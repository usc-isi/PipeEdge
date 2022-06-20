# pylint: disable=C0116
"""Test yaml_types module."""
import unittest
from pipeedge.sched.yaml_types import yaml_device_type, yaml_model, yaml_model_profile

class TestYamlModel(unittest.TestCase):
    """Test yaml_model."""

    def test_yaml_model_empty(self):
        model = yaml_model(0, 0, [], [])
        self.assertIsInstance(model, dict)
        self.assertEqual(model['layers'], 0)
        self.assertEqual(model['parameters_in'], 0)
        self.assertEqual(model['parameters_out'], [])
        self.assertEqual(model['mem_MB'], [])

    def test_yaml_model_single_layer(self):
        model = yaml_model(1, 2, [3], [4])
        self.assertIsInstance(model, dict)
        self.assertEqual(model['layers'], 1)
        self.assertEqual(model['parameters_in'], 2)
        self.assertEqual(model['parameters_out'], [3])
        self.assertEqual(model['mem_MB'], [4])


class TestYamlModeProfile(unittest.TestCase):
    """Test yaml_model_profile."""

    def test_yaml_model_profile(self):
        model_profile = yaml_model_profile('foo', 1, [2])
        self.assertIsInstance(model_profile, dict)
        self.assertEqual(model_profile['dtype'], 'foo')
        self.assertEqual(model_profile['batch_size'], 1)
        self.assertEqual(model_profile['time_s'], [2])


class TestYamlDeviceType(unittest.TestCase):
    """Test yaml_device_type."""

    def test_yaml_device_type(self):
        device_type = yaml_device_type(1, 2, {})
        self.assertIsInstance(device_type, dict)
        self.assertEqual(device_type['mem_MB'], 1)
        self.assertEqual(device_type['bw_Mbps'], 2)
        self.assertEqual(device_type['model_profiles'], {})
