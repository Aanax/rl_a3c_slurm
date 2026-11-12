import os
import unittest


class EmpiricalDistributionCorrectionTest(unittest.TestCase):
    def test_two_level_actor_does_not_rescale_by_action_probability(self):
        train_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'train.py')
        with open(train_path) as train_file:
            source = train_file.read()
        self.assertNotIn('empirical_distribution_correction', source)
        self.assertIn('oracle_two_level_actor_loss(', source)


if __name__ == '__main__':
    unittest.main()
