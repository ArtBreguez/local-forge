import unittest
from solution import maze_steps


class MazeTests(unittest.TestCase):
    def test_example(self):
        self.assertEqual(maze_steps(["S.#", "..#", "#.E"]), 4)

    def test_straight_line(self):
        self.assertEqual(maze_steps(["S...E"]), 4)
        self.assertEqual(maze_steps(["S", ".", ".", "E"]), 3)
        self.assertEqual(maze_steps(["SE"]), 1)

    def test_detour(self):
        grid = ["S.#....",
                ".##.##.",
                "...#...",
                ".#...#E"]
        self.assertEqual(maze_steps(grid), 11)

    def test_unreachable(self):
        self.assertEqual(maze_steps(["S#E"]), -1)
        self.assertEqual(maze_steps(["S.#", "..#", "#.#", "##E"]), -1)

    def test_walls_block_diagonals(self):
        self.assertEqual(maze_steps(["S#", "#E"]), -1)

    def test_shortest_not_first_found(self):
        grid = ["S....",
                ".###.",
                ".###.",
                "....E"]
        self.assertEqual(maze_steps(grid), 7)

    def test_start_in_the_middle(self):
        grid = [".....",
                "..S..",
                ".....",
                "E...."]
        self.assertEqual(maze_steps(grid), 4)

    def test_open_field(self):
        grid = ["S....", ".....", ".....", "....E"]
        self.assertEqual(maze_steps(grid), 7)

    def test_validation(self):
        bad = [["S.", "E"], ["S.E?"], ["S.."], ["E.."], ["SS.E"], ["S.EE"], ["S x E"], ["s.E"], ["S.", "ee"]]
        for grid in bad:
            with self.subTest(grid=grid):
                with self.assertRaises(ValueError):
                    maze_steps(grid)

    def test_input_not_modified(self):
        grid = ["S.#", "..#", "#.E"]
        maze_steps(grid)
        self.assertEqual(grid, ["S.#", "..#", "#.E"])


if __name__ == "__main__":
    unittest.main()
