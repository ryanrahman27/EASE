import unittest
import xml.etree.ElementTree as ET
import mujoco
from examples.ease_fm.pick_place_fixture import run_case, arm_fixture_xml
from examples.ease_fm.mixed_sim import sample_arm


class PickPlaceTest(unittest.TestCase):
    def test_physical_grasp_lift_and_release_without_attachment(self):
        result,xml=run_case()
        self.assertTrue(result['success'])
        self.assertTrue(result['lifted'])
        self.assertFalse(result['forbidden_contact'])
        self.assertIsNone(ET.fromstring(xml).find('equality'))
        self.assertFalse(result['trace'][-1]['bilateral'])

    def test_serial_arm_assets_have_arm_and_jaw_actuators(self):
        for family in ('six','seven'):
            arm=sample_arm(2,family,'fixture');xml=arm_fixture_xml(arm)
            model=mujoco.MjModel.from_xml_string(xml)
            self.assertEqual(model.nu,arm.dof+2)
            self.assertEqual(model.nv,arm.dof+2+6)
            self.assertIsNone(ET.fromstring(xml).find('equality'))


if __name__=='__main__':unittest.main()
