import unittest

from fractional_cascading_layer import FCLayer, FCSearchResult


class TestFCLayerConstruction(unittest.TestCase):
    def test_empty_layer(self):
        layer = FCLayer([])
        self.assertEqual(layer.keys, [])
        self.assertEqual(len(layer), 0)
        self.assertIsNone(layer.next)

    def test_unsorted_input_becomes_sorted(self):
        layer = FCLayer([5, 1, 3, 2, 4])
        self.assertEqual(layer.keys, [1, 2, 3, 4, 5])

    def test_input_is_snapshotted(self):
        # Mutating the original list must not corrupt the layer.
        original = [3, 1, 2]
        layer = FCLayer(original)
        original.append(0)
        original.sort()
        self.assertEqual(layer.keys, [1, 2, 3])

    def test_duplicate_keys_raise(self):
        with self.assertRaises(ValueError):
            FCLayer([1, 1])

    def test_duplicate_after_sort_raise(self):
        with self.assertRaises(ValueError):
            FCLayer([1, 2, 1])

    def test_keys_property_returns_copy(self):
        layer = FCLayer([1, 2, 3])
        k = layer.keys
        k.append(99)
        self.assertEqual(layer.keys, [1, 2, 3])

    def test_next_layer_link(self):
        second = FCLayer([10, 20, 30])
        first = FCLayer([1, 2, 3], next_layer=second)
        self.assertIs(first.next, second)
        self.assertIsNone(second.next)


class TestSearch(unittest.TestCase):
    def test_found_element(self):
        layer = FCLayer([1, 3, 5, 7, 9])
        r = layer.search(5)
        self.assertTrue(r.found)
        self.assertEqual(r.position, 2)
        self.assertIs(r.layer, layer)

    def test_first_element(self):
        layer = FCLayer([1, 3, 5])
        r = layer.search(1)
        self.assertTrue(r.found)
        self.assertEqual(r.position, 0)

    def test_last_element(self):
        layer = FCLayer([1, 3, 5])
        r = layer.search(5)
        self.assertTrue(r.found)
        self.assertEqual(r.position, 2)

    def test_missing_smaller_than_all(self):
        layer = FCLayer([1, 3, 5])
        r = layer.search(0)
        self.assertFalse(r.found)
        self.assertEqual(r.position, 0)

    def test_missing_between_elements(self):
        layer = FCLayer([1, 3, 5, 7])
        r = layer.search(4)
        self.assertFalse(r.found)
        self.assertEqual(r.position, 2)

    def test_missing_larger_than_all(self):
        layer = FCLayer([1, 3, 5])
        r = layer.search(99)
        self.assertFalse(r.found)
        self.assertEqual(r.position, 3)

    def test_search_empty_layer(self):
        layer = FCLayer([])
        r = layer.search(42)
        self.assertFalse(r.found)
        self.assertEqual(r.position, 0)


class TestLocateInNext(unittest.TestCase):
    def test_no_next_returns_none(self):
        layer = FCLayer([1, 2, 3])
        self.assertIsNone(layer.locate_in_next(2))

    def test_found_in_next(self):
        second = FCLayer([10, 20, 30])
        first = FCLayer([1, 2, 3], next_layer=second)
        r = first.locate_in_next(20)
        self.assertTrue(r.found)
        self.assertEqual(r.position, 1)
        self.assertIs(r.layer, second)

    def test_missing_in_next_smaller(self):
        second = FCLayer([10, 20, 30])
        first = FCLayer([1, 2, 3], next_layer=second)
        r = first.locate_in_next(5)
        self.assertFalse(r.found)
        self.assertEqual(r.position, 0)

    def test_missing_in_next_between(self):
        second = FCLayer([10, 20, 30])
        first = FCLayer([1, 2, 3], next_layer=second)
        r = first.locate_in_next(15)
        self.assertFalse(r.found)
        self.assertEqual(r.position, 1)

    def test_missing_in_next_larger(self):
        second = FCLayer([10, 20, 30])
        first = FCLayer([1, 2, 3], next_layer=second)
        r = first.locate_in_next(99)
        self.assertFalse(r.found)
        self.assertEqual(r.position, 3)

    def test_locate_in_empty_next(self):
        second = FCLayer([])
        first = FCLayer([1, 2, 3], next_layer=second)
        r = first.locate_in_next(5)
        self.assertFalse(r.found)
        self.assertEqual(r.position, 0)

    def test_layers_with_no_overlap(self):
        # Keys present in current but absent in next, and vice versa.
        second = FCLayer([10, 20, 30])
        first = FCLayer([1, 2, 3], next_layer=second)
        for x in (1, 2, 3):
            r = first.locate_in_next(x)
            self.assertFalse(r.found)
        for x in (10, 20, 30):
            r = first.locate_in_next(x)
            self.assertTrue(r.found)


class TestChainSearch(unittest.TestCase):
    def test_single_layer_chain(self):
        layer = FCLayer([1, 3, 5])
        results = layer.chain_search(3)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].found)
        self.assertEqual(results[0].position, 1)

    def test_two_layer_chain_found_both(self):
        second = FCLayer([3, 6, 9])
        first = FCLayer([1, 3, 5], next_layer=second)
        results = first.chain_search(3)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].found)
        self.assertEqual(results[0].position, 1)
        self.assertTrue(results[1].found)
        self.assertEqual(results[1].position, 0)

    def test_two_layer_chain_found_first_only(self):
        second = FCLayer([10, 20, 30])
        first = FCLayer([1, 3, 5], next_layer=second)
        results = first.chain_search(3)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].found)
        self.assertFalse(results[1].found)

    def test_two_layer_chain_found_second_only(self):
        second = FCLayer([3, 6, 9])
        first = FCLayer([1, 2, 5], next_layer=second)
        results = first.chain_search(3)
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0].found)
        self.assertEqual(results[0].position, 2)
        self.assertTrue(results[1].found)
        self.assertEqual(results[1].position, 0)

    def test_three_layer_chain(self):
        third = FCLayer([100, 200, 300])
        second = FCLayer([10, 20, 30], next_layer=third)
        first = FCLayer([1, 2, 3], next_layer=second)
        results = first.chain_search(200)
        self.assertEqual(len(results), 3)
        self.assertFalse(results[0].found)
        self.assertFalse(results[1].found)
        self.assertTrue(results[2].found)
        self.assertEqual(results[2].position, 1)

    def test_chain_empty_layers(self):
        third = FCLayer([])
        second = FCLayer([], next_layer=third)
        first = FCLayer([], next_layer=second)
        results = first.chain_search(42)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertFalse(r.found)
            self.assertEqual(r.position, 0)


class TestFCSearchResult(unittest.TestCase):
    def test_repr_contains_position_and_found(self):
        layer = FCLayer([1, 2, 3])
        r = layer.search(2)
        s = repr(r)
        self.assertIn("position=1", s)
        self.assertIn("found=True", s)

    def test_eq_same_values(self):
        layer = FCLayer([1, 2, 3])
        r1 = layer.search(2)
        r2 = layer.search(2)
        self.assertEqual(r1, r2)

    def test_eq_different_position(self):
        layer = FCLayer([1, 2, 3])
        r1 = layer.search(2)
        r2 = layer.search(99)
        self.assertNotEqual(r1, r2)

    def test_eq_different_found(self):
        layer = FCLayer([1, 2, 3])
        r1 = layer.search(2)
        r2 = layer.search(4)
        # position differs too, but found also differs
        self.assertNotEqual(r1, r2)

    def test_eq_different_layer(self):
        a = FCLayer([1, 2, 3])
        b = FCLayer([1, 2, 3])
        r1 = a.search(2)
        r2 = b.search(2)
        self.assertNotEqual(r1, r2)

    def test_eq_against_non_result(self):
        layer = FCLayer([1, 2, 3])
        r = layer.search(2)
        self.assertNotEqual(r, "not a result")
        self.assertNotEqual(r, 42)


class TestNegativesAndBoundaries(unittest.TestCase):
    def test_negative_keys(self):
        layer = FCLayer([-5, -3, -1, 0, 2])
        r = layer.search(-3)
        self.assertTrue(r.found)
        self.assertEqual(r.position, 1)

    def test_negative_missing(self):
        layer = FCLayer([-5, -3, -1])
        r = layer.search(-4)
        self.assertFalse(r.found)
        self.assertEqual(r.position, 1)

    def test_single_element_found(self):
        layer = FCLayer([42])
        r = layer.search(42)
        self.assertTrue(r.found)
        self.assertEqual(r.position, 0)

    def test_single_element_missing_below(self):
        layer = FCLayer([42])
        r = layer.search(1)
        self.assertFalse(r.found)
        self.assertEqual(r.position, 0)

    def test_single_element_missing_above(self):
        layer = FCLayer([42])
        r = layer.search(99)
        self.assertFalse(r.found)
        self.assertEqual(r.position, 1)

    def test_cross_layer_single_element_each(self):
        second = FCLayer([50])
        first = FCLayer([5], next_layer=second)
        r1 = first.search(5)
        self.assertTrue(r1.found)
        r2 = first.locate_in_next(50)
        self.assertTrue(r2.found)
        self.assertEqual(r2.position, 0)


if __name__ == "__main__":
    unittest.main()
