"""
game_2048.py
============

A configurable 2048 game engine with both human (pygame) and programmatic (agent) interfaces.

The module is split into three layers:
    - Game2048: Pure game logic — no rendering, no I/O. Agents interact with this directly.
    - Game2048Visual: Pygame-based renderer that reads from a Game2048 instance.
    - Helper functions: Config loading, human play loop, and agent demo.

Typical agent usage::

    from game_2048 import Game2048, Action
    game = Game2048({'grid_size': 4})
    while not game.is_game_over():
        state = game.get_state()
        action = my_agent.choose(state)
        valid, reward = game.move(action)

Typical MCTS / search agent usage::

    # Clone the game state for simulation — no side effects on the real game
    sim = game.clone()
    sim.move(Action.LEFT)

    # Or construct from a raw board for arbitrary state evaluation
    sim = Game2048.from_state(some_board, score=1200)

Typical human usage::

    python game_2048.py              # reads config.json
    python game_2048.py --agent      # random agent demo
"""

import pygame
import numpy as np
import json
from typing import Tuple, Optional, Union, Dict, List, Literal
from enum import Enum


class Action(Enum):
    """Enum representing the four possible move directions.

    Attributes:
        UP (int): Slide tiles upward (value 0).
        DOWN (int): Slide tiles downward (value 1).
        LEFT (int): Slide tiles to the left (value 2).
        RIGHT (int): Slide tiles to the right (value 3).
    """
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Game2048:
    """Core 2048 game engine — pure logic, no rendering.

    This class manages the board state, move execution, tile spawning,
    scoring, and game-over detection. It exposes a clean API for agents
    to query state, execute actions, and receive rewards.

    Supports cheap cloning via ``clone()`` and ``from_state()`` so that
    search-based agents (MCTS, expectimax, minimax, etc.) can simulate
    moves without affecting the real game.

    Attributes:
        grid_size (int): Side length of the square board (e.g. 4 for 4×4).
        tile_2_prob (float): Probability that a new tile is a 2 (vs 4).
        initial_tiles (int): Number of tiles placed at game start.
        seed (int | None): Random seed for reproducibility. None = non-deterministic.
        board (np.ndarray): Current board state, shape (grid_size, grid_size), dtype int.
        score (int): Cumulative score (sum of all merged tile values).
        game_over (bool): Whether the game has ended (no valid moves remain).

    Args:
        config (Dict): Configuration dictionary with optional keys:
            - 'grid_size' (int, default 4): Board dimension.
            - 'tile_2_probability' (float, default 0.9): Probability of spawning a 2.
            - 'initial_tiles' (int, default 2): Tiles placed at start.
            - 'random_seed' (int | None, default None): RNG seed.

    Example::

        game = Game2048({'grid_size': 4, 'random_seed': 42})
        state = game.get_state()       # np.ndarray (4, 4)
        valid, reward = game.move(Action.LEFT)
    """

    def __init__(self, config: Dict):
        self.grid_size = config.get('grid_size', 4)
        self.tile_2_prob = config.get('tile_2_probability', 0.9)
        self.initial_tiles = config.get('initial_tiles', 2)
        self.seed = config.get('random_seed', None)

        if self.seed is not None:
            np.random.seed(self.seed)

        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.score = 0
        self.game_over = False

        for _ in range(self.initial_tiles):
            self._add_random_tile()

    # ─── Cloning / state construction ─────────────────────────────

    @classmethod
    def from_state(cls, board: np.ndarray, score: int = 0,
                   config: Optional[Dict] = None) -> 'Game2048':
        """Create a Game2048 instance from an existing board state.

        Useful for search-based agents that need to evaluate arbitrary
        board positions without playing through a full game. The returned
        instance is fully functional — you can call ``move()``,
        ``get_available_moves()``, ``is_game_over()``, etc.

        The instance's RNG is **not** seeded, so tile spawns during
        simulation are non-deterministic (appropriate for MCTS rollouts).

        Args:
            board (np.ndarray): 2D integer array representing the board.
                Shape determines ``grid_size`` if config is not provided.
            score (int): Starting score. Defaults to 0.
            config (Dict | None): Optional config overrides. If None,
                defaults are used with ``grid_size`` inferred from ``board``.

        Returns:
            Game2048: A new instance with the given board and score.

        Example::

            import numpy as np
            board = np.array([[2, 4, 0, 0],
                              [0, 2, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 2]])
            game = Game2048.from_state(board, score=100)
            game.move(Action.LEFT)
        """
        config = config or {}
        obj = cls.__new__(cls)
        obj.grid_size = config.get('grid_size', board.shape[0])
        obj.tile_2_prob = config.get('tile_2_probability', 0.9)
        obj.initial_tiles = config.get('initial_tiles', 2)
        obj.seed = None  # simulations use independent RNG
        obj.board = board.copy()
        obj.score = score
        obj.game_over = False
        return obj

    def clone(self) -> 'Game2048':
        """Create an independent copy of this game for simulation.

        The clone shares no mutable state with the original — modifying
        the clone (calling ``move()``, etc.) has zero side effects on
        the source instance. The clone's RNG is independent.

        This is the primary method search agents should use::

            for action in game.get_available_moves():
                sim = game.clone()
                sim.move(action)
                value = evaluate(sim)

        Returns:
            Game2048: A deep copy with identical board, score, and config
                but independent state.
        """
        return Game2048.from_state(
            board=self.board,
            score=self.score,
            config={
                'grid_size': self.grid_size,
                'tile_2_probability': self.tile_2_prob,
                'initial_tiles': self.initial_tiles,
            },
        )

    # ─── Core game logic ──────────────────────────────────────────

    def _add_random_tile(self) -> bool:
        """Place a new tile (2 or 4) on a random empty cell.

        The tile value is determined by ``tile_2_prob``: with that probability
        the tile is a 2, otherwise a 4. The cell is chosen uniformly at random
        from all empty cells.

        Returns:
            bool: True if a tile was placed, False if the board is full.
        """
        empty = list(zip(*np.where(self.board == 0)))
        if not empty:
            return False
        r, c = empty[np.random.randint(len(empty))]
        self.board[r, c] = 2 if np.random.random() < self.tile_2_prob else 4
        return True

    def get_state(self) -> np.ndarray:
        """Return a copy of the current board state.

        Returns:
            np.ndarray: Board matrix of shape (grid_size, grid_size) with
                integer tile values. Empty cells are 0.
        """
        return self.board.copy()

    def get_score(self) -> int:
        """Return the current cumulative score.

        Returns:
            int: Total points accumulated from all merges so far.
        """
        return self.score

    def is_game_over(self) -> bool:
        """Check whether the game has ended.

        The game is over when the board is full and no adjacent tiles
        can merge (horizontally or vertically).

        Returns:
            bool: True if no valid moves remain, False otherwise.
        """
        if self.game_over:
            return True
        if np.any(self.board == 0):
            return False
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                val = self.board[r, c]
                if c + 1 < self.grid_size and val == self.board[r, c + 1]:
                    return False
                if r + 1 < self.grid_size and val == self.board[r + 1, c]:
                    return False
        self.game_over = True
        return True

    def get_available_moves(self) -> List[Action]:
        """Return a list of actions that would change the board state.

        Each candidate action is simulated on a copy of the board. Only
        actions that produce a different board configuration are included.

        Returns:
            List[Action]: Valid actions from the current state. May be empty
                if the game is over.
        """
        moves = []
        for action in Action:
            board_copy = self.board.copy()
            score_copy = self.score
            changed, _ = self._execute_move(action)
            if changed:
                moves.append(action)
            self.board = board_copy
            self.score = score_copy
        return moves

    def calculate_reward(self, old_board: np.ndarray, new_board: np.ndarray,
                         merges: List[Tuple[int, int, int]]) -> int:
        """Calculate the reward for a completed move.

        This method is called after every valid move and can be overridden
        in subclasses to implement custom reward shaping for agent training.

        Args:
            old_board (np.ndarray): Board state before the move.
            new_board (np.ndarray): Board state after the move (before new tile spawn).
            merges (List[Tuple[int, int, int]]): List of merge events, each a tuple
                of (tile_a_value, tile_b_value, merged_value). For example,
                two 4-tiles merging produces (4, 4, 8).

        Returns:
            int: Reward value. Default implementation returns the sum of all
                merged tile values (standard 2048 scoring).

        Example override::

            class Custom2048(Game2048):
                def calculate_reward(self, old_board, new_board, merges):
                    # Reward = number of empty cells after move
                    return int(np.sum(new_board == 0))
        """
        return sum(val for _, _, val in merges)

    def _slide_and_merge_line(self, line: np.ndarray) -> Tuple[np.ndarray, int, List[Tuple[int, int, int]]]:
        """Slide a single row or column toward index 0, merging equal adjacent tiles.

        This is the core merge algorithm. It processes one 1D slice of the board:
            1. Remove zeros (compact non-empty tiles).
            2. Walk left-to-right; if two adjacent tiles are equal, merge them
               into one tile of double value. Each tile can only merge once per move.
            3. Pad with zeros on the right to restore original length.

        Args:
            line (np.ndarray): 1D array representing a single row or column.

        Returns:
            Tuple containing:
                - np.ndarray: The resulting line after slide and merge.
                - int: Points scored from merges in this line.
                - List[Tuple[int, int, int]]: Merge events as
                  (tile_a, tile_b, merged_value) tuples.

        Example::

            # [2, 2, 4, 4] -> [4, 8, 0, 0], points=12, merges=[(2,2,4),(4,4,8)]
        """
        non_zero = line[line != 0]
        merged = []
        merge_list = []
        points = 0
        skip = False

        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                val = non_zero[i] * 2
                merged.append(val)
                merge_list.append((int(non_zero[i]), int(non_zero[i + 1]), int(val)))
                points += val
                skip = True
            else:
                merged.append(non_zero[i])

        result = np.zeros_like(line)
        for i, v in enumerate(merged):
            result[i] = v
        return result, points, merge_list

    def _execute_move(self, action: Action) -> Tuple[bool, List[Tuple[int, int, int]]]:
        """Execute a move on the board in-place.

        Applies the slide-and-merge operation to every row or column in the
        direction specified by ``action``. Updates ``self.board`` and
        ``self.score`` if the board changes.

        Args:
            action (Action): The direction to slide tiles.

        Returns:
            Tuple containing:
                - bool: True if the board changed (move was valid).
                - List[Tuple[int, int, int]]: All merge events from this move.
        """
        original = self.board.copy()
        total_points = 0
        all_merges = []

        if action == Action.LEFT:
            for r in range(self.grid_size):
                self.board[r], pts, merges = self._slide_and_merge_line(self.board[r])
                total_points += pts
                all_merges.extend(merges)

        elif action == Action.RIGHT:
            for r in range(self.grid_size):
                self.board[r], pts, merges = self._slide_and_merge_line(self.board[r][::-1])
                self.board[r] = self.board[r][::-1]
                total_points += pts
                all_merges.extend(merges)

        elif action == Action.UP:
            for c in range(self.grid_size):
                col = self.board[:, c]
                self.board[:, c], pts, merges = self._slide_and_merge_line(col)
                total_points += pts
                all_merges.extend(merges)

        elif action == Action.DOWN:
            for c in range(self.grid_size):
                col = self.board[:, c][::-1]
                merged, pts, merges = self._slide_and_merge_line(col)
                self.board[:, c] = merged[::-1]
                total_points += pts
                all_merges.extend(merges)

        changed = not np.array_equal(original, self.board)
        if changed:
            self.score += total_points
        return changed, all_merges

    def move(self, action: Action) -> Tuple[bool, int]:
        """Execute a move and spawn a new tile if valid.

        This is the primary method agents should call. It:
            1. Executes the slide-and-merge for the given direction.
            2. Computes reward via ``calculate_reward()``.
            3. Spawns a new random tile.
            4. Checks for game over.

        Args:
            action (Action): The direction to slide tiles.

        Returns:
            Tuple containing:
                - bool: True if the move was valid (board changed), False otherwise.
                - int: Reward for this move (0 if invalid).
        """
        if self.game_over:
            return False, 0

        old_board = self.board.copy()
        changed, merges = self._execute_move(action)

        if not changed:
            return False, 0

        reward = self.calculate_reward(old_board, self.board, merges)
        self._add_random_tile()
        self.is_game_over()
        return True, reward

    def reset(self):
        """Reset the game to a fresh initial state.

        Clears the board, resets score to 0, and places ``initial_tiles``
        new random tiles. Does NOT reset the random seed, so subsequent
        games will continue from the current RNG state.
        """
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.score = 0
        self.game_over = False
        for _ in range(self.initial_tiles):
            self._add_random_tile()


class Game2048Visual:
    """Pygame-based visual renderer for a Game2048 instance.

    This class handles all rendering and keyboard input. It does NOT
    modify game logic — it only reads from the ``Game2048`` instance
    to draw the board and delegates moves back to the game object.

    Attributes:
        game (Game2048): The game instance to render.
        cell (int): Pixel size of each tile cell.
        pad (int): Pixel padding between cells.
        w (int): Window width in pixels.
        h (int): Window height in pixels (includes score bar).
        screen (pygame.Surface): The pygame display surface.

    Args:
        game (Game2048): Game instance to visualize.
        config (Dict): Configuration dictionary with optional keys:
            - 'cell_size' (int, default 100): Pixel size per tile.
            - 'cell_padding' (int, default 10): Pixel gap between tiles.
    """

    TILE_COLORS = {
        0: (205, 193, 180),
        2: (238, 228, 218),
        4: (237, 224, 200),
        8: (242, 177, 121),
        16: (245, 149, 99),
        32: (246, 124, 95),
        64: (246, 94, 59),
        128: (237, 207, 114),
        256: (237, 204, 97),
        512: (237, 200, 80),
        1024: (237, 197, 63),
        2048: (237, 194, 46),
    }
    """Dict[int, Tuple[int, int, int]]: RGB color mapping for tile values."""

    DEFAULT_TILE_COLOR = (60, 58, 50)
    """Tuple[int, int, int]: Fallback color for tiles > 2048."""

    BG_COLOR = (187, 173, 160)
    """Tuple[int, int, int]: Background color of the game board."""

    def __init__(self, game: Game2048, config: Dict):
        pygame.init()
        self.game = game
        self.cell = config.get('cell_size', 100)
        self.pad = config.get('cell_padding', 10)
        n = game.grid_size
        self.w = (self.cell + self.pad) * n + self.pad
        self.h = self.w + 80
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption(f'2048 ({n}x{n})')
        self.font = pygame.font.Font(None, max(20, 60 - n * 4))
        self.score_font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 48)

    def _tile_rect(self, r: int, c: int) -> pygame.Rect:
        """Compute the pixel rectangle for a tile at board position (r, c).

        Args:
            r (int): Row index (0-based, top to bottom).
            c (int): Column index (0-based, left to right).

        Returns:
            pygame.Rect: Rectangle for the tile's screen position and size.
        """
        x = self.pad + c * (self.cell + self.pad)
        y = 80 + self.pad + r * (self.cell + self.pad)
        return pygame.Rect(x, y, self.cell, self.cell)

    def draw(self):
        """Render the current game state to the pygame window.

        Draws the background, score display, game-over overlay (if applicable),
        all tiles with value-dependent colors, and a controls hint bar.
        Calls ``pygame.display.flip()`` to push the frame.
        """
        self.screen.fill(self.BG_COLOR)

        # Score
        txt = self.score_font.render(f'Score: {self.game.score}', True, (255, 255, 255))
        self.screen.blit(txt, (20, 20))

        # Game over
        if self.game.game_over:
            go = self.big_font.render('GAME OVER', True, (255, 0, 0))
            self.screen.blit(go, go.get_rect(center=(self.w // 2, 50)))

        # Tiles
        for r in range(self.game.grid_size):
            for c in range(self.game.grid_size):
                val = self.game.board[r, c]
                rect = self._tile_rect(r, c)
                color = self.TILE_COLORS.get(val, self.DEFAULT_TILE_COLOR)
                pygame.draw.rect(self.screen, color, rect, border_radius=6)

                if val != 0:
                    tc = (119, 110, 101) if val <= 4 else (255, 255, 255)
                    num = self.font.render(str(val), True, tc)
                    self.screen.blit(num, num.get_rect(center=rect.center))

        # Controls hint
        hint = pygame.font.Font(None, 22).render(
            'Arrow/WASD to move | R to reset | Q to quit', True, (255, 255, 255)
        )
        self.screen.blit(hint, hint.get_rect(center=(self.w // 2, self.h - 15)))

        pygame.display.flip()

    def handle_input(self) -> Union[Action, Literal['reset', 'quit'], None]:
        """Process pygame events and return the corresponding action.

        Supports arrow keys and WASD for movement, R for reset, Q for quit.

        Returns:
            Action: If a movement key was pressed.
            Literal['reset']: If R was pressed.
            Literal['quit']: If Q was pressed or the window was closed.
            None: If no relevant event occurred this frame.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            if event.type == pygame.KEYDOWN:
                mapping = {
                    pygame.K_UP: Action.UP, pygame.K_w: Action.UP,
                    pygame.K_DOWN: Action.DOWN, pygame.K_s: Action.DOWN,
                    pygame.K_LEFT: Action.LEFT, pygame.K_a: Action.LEFT,
                    pygame.K_RIGHT: Action.RIGHT, pygame.K_d: Action.RIGHT,
                }
                if event.key in mapping:
                    return mapping[event.key]
                if event.key == pygame.K_r:
                    return 'reset'
                if event.key == pygame.K_q:
                    return 'quit'
        return None

    def close(self):
        """Shut down the pygame display and clean up resources."""
        pygame.quit()


# ─── Config loader ────────────────────────────────────────────────

def load_config(filepath: Optional[str] = None) -> Dict:
    """Load game configuration from a JSON file, falling back to defaults.

    Default configuration::

        {
            "grid_size": 4,
            "tile_2_probability": 0.9,
            "initial_tiles": 2,
            "random_seed": null,
            "cell_size": 100,
            "cell_padding": 10
        }

    Any keys present in the JSON file override the corresponding defaults.

    Args:
        filepath: Path to a JSON config file. If None or
            the file doesn't exist, all defaults are used.

    Returns:
        Merged configuration dictionary.
    """
    defaults: Dict = {
        'grid_size': 4,
        'tile_2_probability': 0.9,
        'initial_tiles': 2,
        'random_seed': None,
        'cell_size': 100,
        'cell_padding': 10,
    }
    if filepath:
        try:
            with open(filepath, 'r') as f:
                defaults.update(json.load(f))
        except FileNotFoundError:
            print(f"Config {filepath} not found, using defaults.")
    return defaults


# ─── Human play ───────────────────────────────────────────────────

def play_game_human(config_path: Optional[str] = None):
    """Launch a human-playable 2048 game with pygame visualization.

    Opens a pygame window and runs the game loop. The player uses
    arrow keys or WASD to move tiles, R to reset, and Q to quit.

    Args:
        config_path (str | None): Path to a JSON config file.
            Passed to ``load_config()``.
    """
    config = load_config(config_path)
    game = Game2048(config)
    vis = Game2048Visual(game, config)
    clock = pygame.time.Clock()

    vis.draw()
    running = True
    while running:
        result = vis.handle_input()
        if result == 'quit':
            running = False
        elif result == 'reset':
            game.reset()
            vis.draw()
        elif isinstance(result, Action):
            game.move(result)
            vis.draw()
        clock.tick(60)

    vis.close()


# ─── Agent example ────────────────────────────────────────────────

def example_agent_play(config_path: Optional[str] = None):
    """Run a random agent demo and print each step to stdout.

    Loads configuration from the given JSON file (or defaults) and
    plays random valid moves until game over, printing the action,
    reward, and score at each step.

    Args:
        config_path (str | None): Path to a JSON config file.
            Passed to ``load_config()``.
    """
    config = load_config(config_path)
    game = Game2048(config)

    while not game.is_game_over():
        available = game.get_available_moves()
        if not available:
            break
        action = available[np.random.randint(len(available))]
        valid, reward = game.move(action)
        print(f"Action: {action.name}  Reward: {reward}  Score: {game.score}")

    print(f"\nFinal Score: {game.score}")
    print(game.get_state())


if __name__ == "__main__":
    import sys
    if '--agent' in sys.argv:
        example_agent_play('config.json')
    else:
        play_game_human('config.json')