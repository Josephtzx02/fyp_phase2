#testopen3d.py
#import open3d as o3d
#print(o3d.__version__)

from robot_world import RobotWorld

world = RobotWorld()

world.create_world()

world.run()