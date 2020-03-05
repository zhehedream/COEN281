import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import Geometric
from diffprivlib.mechanisms.transforms import StringToInt
from diffprivlib.utils import global_seed

global_seed(3141592653)


class TestStringToInt(TestCase):
    def test_not_none(self):
        mech = StringToInt(Geometric())
        self.assertIsNotNone(mech)
        _mech = mech.copy()
        self.assertIsNotNone(_mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMachine
        self.assertTrue(issubclass(StringToInt, DPMachine))

    def test_no_parent(self):
        with self.assertRaises(TypeError):
            StringToInt()

    def test_empty_mechanism(self):
        mech = StringToInt(Geometric())
        with self.assertRaises(ValueError):
            mech.randomise("1")

    def test_set_epsilon_locally(self):
        mech = StringToInt(Geometric().set_sensitivity(1))
        mech.set_epsilon(1)
        self.assertIsNotNone(mech)

    def test_randomise(self):
        mech = StringToInt(Geometric().set_sensitivity(1).set_epsilon(1))
        self.assertIsInstance(mech.randomise("1"), str)

    def test_distrib(self):
        epsilon = 1.0
        runs = 10000
        mech = StringToInt(Geometric().set_sensitivity(1).set_epsilon(epsilon))
        count = [0, 0]

        for _ in range(runs):
            if mech.randomise("0") == "0":
                count[0] += 1

            if mech.randomise("1") == "0":
                count[1] += 1

        self.assertGreater(count[0], count[1])
        # print("%f <= %f" % (count[0] / runs, count[1] * np.exp(epsilon) / runs))
        self.assertLessEqual(count[0] / runs, count[1] * np.exp(epsilon) / runs + 0.05)
