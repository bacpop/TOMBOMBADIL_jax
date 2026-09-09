import os
import tempfile
import unittest

from plot_param_estimates import find_scalar_csvs, load_scalar_csv


class TestScalarParamPlots(unittest.TestCase):
    def test_load_scalar_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            scalar_path = os.path.join(tmp, "scalar_fit_Allparams.csv")
            with open(scalar_path, "w") as f:
                f.write("variable,value\nalpha,1.0\nomega,0.5\n")

            params = load_scalar_csv(scalar_path)

        self.assertEqual(params["alpha"], 1.0)
        self.assertEqual(params["omega"], 0.5)

    def test_find_scalar_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            scalar_path = os.path.join(tmp, "scalar_fit_Allparams.csv")
            other_path = os.path.join(tmp, "notes.txt")
            with open(scalar_path, "w") as f:
                f.write("variable,value\nomega,0.5\n")
            with open(other_path, "w") as f:
                f.write("not a scalar output\n")

            paths = find_scalar_csvs(tmp)

        self.assertEqual(paths, [scalar_path])


if __name__ == "__main__":
    unittest.main()
