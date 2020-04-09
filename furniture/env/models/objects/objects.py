import copy
import xml.etree.ElementTree as ET
import numpy as np

from furniture.env.models.base import MujocoXML
from furniture.env.mjcf_utils import string_to_array, array_to_string


class MujocoObject:
    """
    Base class for all objects.

    We use Mujoco Objects to implement all objects that
        1) may appear for multiple times in a task
        2) can be swapped between different tasks

    Typical methods return copy so the caller can all joints/attributes as wanted

    Attributes:
        asset (TYPE): Description
    """

    def __init__(self):
        self.asset = ET.Element("asset")

    def get_bottom_offset(self):
        """
        Returns vector from object center to object bottom
        Helps us put objects on a surface

        Returns:
            np.array: eg. np.array([0, 0, -2])

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError

    def get_top_offset(self):
        """
        Returns vector from object center to object top
        Helps us put other objects on this object

        Returns:
            np.array: eg. np.array([0, 0, 2])

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError

    def get_horizontal_radius(self):
        """
        Returns scalar
        If object a,b has horizontal distance d
        a.get_horizontal_radius() + b.get_horizontal_radius() < d
        should mean that a, b has no contact

        Helps us put objects programmatically without them flying away due to
        a huge initial contact force

        Returns:
            Float: radius

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError
        # return 2

    def get_collision(self, name=None, site=False):
        """
        Returns a ET.Element
        It is a <body/> subtree that defines all collision related fields
        of this object.

        Return is a copy

        Args:
            name (None, optional): Assign name to body
            site (False, optional): Add a site (with name @name
                 when applicable) to the returned body

        Returns:
            ET.Element: body

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError

    def get_visual(self, name=None, site=False):
        """
        Returns a ET.Element
        It is a <body/> subtree that defines all visualization related fields
        of this object.

        Return is a copy

        Args:
            name (None, optional): Assign name to body
            site (False, optional): Add a site (with name @name
                 when applicable) to the returned body

        Returns:
            ET.Element: body

        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError

    def get_site_attrib_template(self):
        """
        Returns attribs of spherical site used to mark body origin

        Returns:
            Dictionary of default site attributes
        """
        return {
            "pos": "0 0 0",
            "size": "0.002 0.002 0.002",
            "rgba": "1 0 0 1",
            "type": "sphere",
        }


class MujocoXMLObject(MujocoXML, MujocoObject):
    """
    MujocoObjects that are loaded from xml files
    """

    def __init__(self, fname, debug=False):
        """
        Args:
            fname (TYPE): XML File path
        """
        MujocoXML.__init__(self, fname, debug)

    def get_bottom_offset(self, name=None):
        if name is None:
            name = self.name
        bottom_site = self.worldbody.find("./body/site[@name='%s_bottom_site']" % name)
        return string_to_array(bottom_site.get("pos"))

    def get_top_offset(self, name=None):
        if name is None:
            name = self.name
        top_site = self.worldbody.find("./body/site[@name='%s_top_site']" % name)
        return string_to_array(top_site.get("pos"))

    def get_horizontal_radius(self, name=None):
        if name is None:
            name = self.name
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='%s_horizontal_radius_site']" % name
        )
        return float(horizontal_radius_site.get("size"))

    def get_collision(self, name=None, site=False, friction=(1, .5, .5)):
        #collision = copy.deepcopy(self.worldbody.find("./body/body[@name='collision']"))
        self.name = name
        collision = copy.deepcopy(self.worldbody.find("./body[@name='%s']" % name))
        collision.attrib.pop("name")
        if name is not None:
            collision.attrib["name"] = name
            geoms = collision.findall("geom")
            for i in range(len(geoms)):
                if not geoms[i].get("name").startswith('noviz'):
                    geoms[i].set("name", "{}-{}".format(name, i))
                geoms[i].set("friction",array_to_string(friction))
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            template["rgba"] = "1 0 0 0"
            if name is not None:
                template["name"] = name
            collision.append(ET.Element("site", attrib=template))
        return collision

    def get_visual(self, name=None, site=False):
        visual = copy.deepcopy(self.worldbody.find("./body/body[@name='visual']"))
        visual.attrib.pop("name")
        if name is not None:
            visual.attrib["name"] = name
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            template["rgba"] = "1 0 0 0"
            if name is not None:
                template["name"] = name
            visual.append(ET.Element("site", attrib=template))
        return visual

