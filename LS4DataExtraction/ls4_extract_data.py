import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import time
import numpy as np
import cv2
import argparse


IM_WIDTH, IM_HEIGHT = 640, 480

def process_img(image, name, vehicle_id):
    print("Frame: "+str(image.frame)+", timestamp: "+str(image.timestamp))
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow(name, i3)
    cv2.waitKey(1)

    if image.frame % 20 == 0:
        image.save_to_disk('_out/vechile_%06d/%06d.png' % (vehicle_id,image.frame))
    return i3/255.0


def main():
    try:
        argparser = argparse.ArgumentParser(
            description=__doc__)
        argparser.add_argument(
            '-n', '--num_vehicles',
            default='1',
            help='Number of vehicles to spawn')
        '''
        argparser.add_argument(
            '-size', '--image-size',
            nargs='+',
            type=int,
            default= '(640,480)',
            help='Size of the image.'
        )
        '''
        args = argparser.parse_args()

        num_vechiles = 1
        #IM_WIDTH, IM_HEIGHT = tuple(args.image_size)

        actor_list = []
        vehicle_list = []
        sensor_list = []

        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)

        world = client.get_world()

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.*.*'))

        # Blueprint for camera
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        cam_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        cam_bp.set_attribute('fov', '110')

        spawn_points = world.get_map().get_spawn_points()
        spawn_point_cam = carla.Transform(carla.Location(x=2.5, z=1.0))

        #Spawn vechiles
        for i in range(num_vechiles):
            vehicle_bp_i = random.choice(blueprint_library.filter('vehicle.*.*'))
            spawn_point_i = spawn_points[i]

            vehicle_i = world.spawn_actor(vehicle_bp_i, spawn_point_i)
            vehicle_i.set_autopilot(True)

            actor_list.append(vehicle_i)
            vehicle_list.append(vehicle_i)

            # Spawn sensor
            camera_i = world.spawn_actor(cam_bp, spawn_point_cam, attach_to=vehicle_i)

            actor_list.append(camera_i)
            sensor_list.append(camera_i)

            # Listen to camera data
            camera_i.listen(lambda data: process_img(data, "camera " + str(i), vehicle_i.id))
        
        
        time.sleep(60)
       # while True:
        #    world.tick()

    finally:
        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('done.')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('Done')