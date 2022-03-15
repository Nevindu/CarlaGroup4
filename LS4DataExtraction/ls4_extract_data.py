''' 
CARLA Group - Team 4
Sensing and Prediction Learing Task 4: Vehicle Data Extraction and Storage
Author: Nevindu M. Batagoda
'''

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
import numpy as np
import cv2
import argparse
import traceback
import csv


def process_img(image, name, vehicle_id, W, H):
    try:
        print(f"VID:{vehicle_id}, Frame: {image.frame}, timestamp: {image.timestamp}")
        i = np.array(image.raw_data)
        i2 = i.reshape((H, W, 4))
        i3 = i2[:, :, :3]
        cv2.imshow(name, i3)
        cv2.waitKey(1)

        if image.frame % 10 == 0:
            image.save_to_disk('_out/vechicle_%06d/%06d.png' % (vehicle_id,image.frame))
        return i3/255.0
    except Exception as e:
        print(traceback.format_exc())
    finally:
        return -1

def writeCSV(header, data):
    with open('_out/actor_info.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def main():
    
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-n', '--num_vehicles',
        metavar='N',
        action='store',
        default=1,
        type=int,
        help='Number of vehicles to spawn')
    argparser.add_argument(
        '-size', '--image_size',
        metavar= '(WIDTH, HEIGHT)',
        nargs=2,
        type=int,
        default= (640,480),
        help='Size of the image.'
    )
    argparser.add_argument(
        '-m', '--map',
        metavar= 'MAP_NAME',
        type=str,
        default= 'Town01',
        help='Name of Map to Load'
    )
    argparser.add_argument(
        '-t', '--run_time',
        metavar= 'T',
        type=int,
        default= None,
        help='How long (in seconds) shoud the simulation run. If not set, the simulation will continue until interrupted.'
    )

    args = vars(argparser.parse_args())

    num_vechiles = args['num_vehicles']
    IM_WIDTH, IM_HEIGHT = tuple(args['image_size'])
    map = args['map']
    t_end = args['run_time']
    if (t_end != None and t_end <= 0):
        raise ValueError("Invalid Simulation End Time.")

    actor_list = []
    vehicle_list = []
    sensor_list = []

    data_dump = []
    csv_header = ['Frame', 'TimeStamp', 'Vehicle_ID', 'Location_X', 'Location_Y',
                 'Velocity_X', 'Velocity_Y', 'Acceleration_X', 'Acceleration_Y']

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

     # Set up the traffic manager
    traffic_manager = client.get_trafficmanager(8000)

    try:    
        world = client.get_world()
        # load map
        client.load_world(map)

        # Set client - server to be synchronous
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05 # 0.05 s in between ticks
        
        traffic_manager.set_synchronous_mode(True)
        # Maybe turn on hybrid mode?

        world.apply_settings(settings)

        # Blueprint for Vehchicle and Camera
        blueprint_library = world.get_blueprint_library()
    
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        cam_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        cam_bp.set_attribute('fov', '110')

        spawn_point_cam = carla.Transform(carla.Location(x=2.5, z=1.0))

        #Spawn vechiles and Sensors
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch_spawn_vehicles = []
        batch_spawn_sensors = []
        for i in range(num_vechiles):
            vehicle_bp_i = random.choice(blueprint_library.filter('vehicle.*.*'))
            spawn_point_i = random.choice(world.get_map().get_spawn_points())
            batch_spawn_vehicles.append(SpawnActor(vehicle_bp_i, spawn_point_i).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
            
        for response in client.apply_batch_sync(batch_spawn_vehicles, True): # Do not tick after spawning vehicles
            if response.error:
                print("Error Spawning Vehicle")
            else:
                vehicle_i = world.get_actor(response.actor_id)
                actor_list.append(vehicle_i)
                vehicle_list.append(vehicle_i)
                batch_spawn_sensors.append(SpawnActor(cam_bp, spawn_point_cam, vehicle_i))

    
        for i, response in enumerate(client.apply_batch_sync(batch_spawn_sensors, True)): # Tick after the sensors are spawned
            if response.error:
                print("Error Spawning Camera Sensor")
            else:
                sensor_i = world.get_actor(response.actor_id)
                actor_list.append(sensor_i)
                sensor_list.append(sensor_i)
                sensor_i.listen(lambda data, n = i: process_img(data, f"camera {n+1}", vehicle_list[n].id, IM_WIDTH, IM_HEIGHT))

        sim_start_time = world.get_snapshot().timestamp.platform_timestamp
        while True:
            world_snapshot = world.get_snapshot()
            frame = world_snapshot.frame
            timestamp = world_snapshot.timestamp.elapsed_seconds
            sim_curr_time = world_snapshot.timestamp.platform_timestamp
            #print(f"Time: {timestamp}, OSTIME:{sim_curr_time}")
            if t_end != None and (sim_curr_time - sim_start_time) > t_end : # End Simulation if reached desired duration
                break 
            for vehicle in vehicle_list:
                vehicle_location = vehicle.get_location()
                vehicle_velocity = vehicle.get_velocity()
                vehicle_aceleation = vehicle.get_acceleration()
                vehicle_id = vehicle.id
                data_dump.append([frame, timestamp, vehicle_id, vehicle_location.x, vehicle_location.y,
                vehicle_velocity.x, vehicle_velocity.y, vehicle_aceleation.x, vehicle_aceleation.y])
            world.tick()

    except Exception as e:
        print(traceback.format_exc())
    finally:
        settings = world.get_settings()
        settings.fixed_delta_seconds = None
        settings.synchronous_mode = False
        traffic_manager.set_synchronous_mode(False)
        world.apply_settings(settings)

        writeCSV(csv_header, data_dump)
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