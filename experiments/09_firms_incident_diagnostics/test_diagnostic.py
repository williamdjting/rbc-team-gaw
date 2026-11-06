import unittest

import h3
import numpy as np
import pandas as pd

from run_diagnostic import chord_km, concentration, connected_labels, sphere


class DiagnosticTests(unittest.TestCase):
    def test_equal_weights_and_concentration(self):
        equal = concentration([1, 1, 1, 1])
        self.assertEqual(equal['effective_cells'], 4)
        self.assertEqual(equal['effective_fraction'], 1)
        self.assertLess(concentration([97, 1, 1, 1])['effective_cells'], 2)

    def test_great_circle_distance(self):
        xyz = sphere([0, 0], [0, 1])
        self.assertAlmostEqual(float(chord_km(np.linalg.norm(xyz[0]-xyz[1]))), 111.19508, places=3)

    def test_components_join_neighbours_and_respect_day_gap(self):
        cell = h3.latlng_to_cell(50.67, -120.33, 7)
        near = next(c for c in h3.grid_disk(cell, 1) if c != cell)
        far = h3.latlng_to_cell(54, -127, 7)
        frame = pd.DataFrame({'day': [1, 1, 4, 10, 1], 'h3_cell': [cell, near, cell, cell, far]})
        one, three = connected_labels(frame, 1), connected_labels(frame, 3)
        self.assertEqual(one[0], one[1])
        self.assertNotEqual(one[0], one[2])
        self.assertEqual(three[0], three[2])
        self.assertNotEqual(three[0], three[3])
        self.assertNotEqual(three[0], three[4])

    def test_transitive_links_can_span_more_than_gap(self):
        cell = h3.latlng_to_cell(50.67, -120.33, 7)
        frame = pd.DataFrame({'day': [7, 1, 4], 'h3_cell': [cell]*3})
        self.assertEqual(len(set(connected_labels(frame, 3))), 1)


if __name__ == '__main__':
    unittest.main()
