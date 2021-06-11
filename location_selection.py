import numpy as np

def grid_round(x, prec=2, base=0.25):     return (base * (np.array(x) / base).round()).round(prec)


def print_location_list(locations):
    print('[', end='')
    for i, loc in enumerate(locations): 
        print('(', end ='')
        print(*loc, sep=',', end='')
        print(')', end = '')
        if not i == len(locations)-1: print(',', end='')
    print(']')
    

def generate_training_locations(n=10, lat_range = [65, 30.25], lon_range = [-20, 20.25]):
    xy_min = [lat_range[0], lon_range[0]]
    xy_max = [lat_range[1], lon_range[1]]
    locations = np.random.uniform(low=xy_min, high=xy_max, size=(n,2))#TODO not same locations
    locations = grid_round(locations, base=0.25)
    locations = [(lat,lon) for lat,lon in locations]
    
    #Print output to copy here in case new random locations are to be defined - maybe later write out as csv? 
    print_location_list(locations)
    return(locations)

#training_locations = generate_training_locations(n=1000)
# n = 10:
training_locations10 = [(56.5,8.5),(44.5,-2.75),(40.25,-2.25),(43.5,12.75),(48.0,1.25),(52.0,6.75),(58.75,16.25),(58.75,16.75),(54.75,-2.5),(46.5,-5.75)]
#n = 100:
training_locations100 = [(49.75,14.5),(32.5,5.25),(38.0,-1.25),(60.0,-3.0),(49.25,8.0),(61.75,-11.5),(56.0,-6.0),(36.25,-19.5),(38.25,6.75),(38.75,15.75),(39.25,-18.0),(35.0,-5.5),(48.0,17.25),(64.5,1.75),(60.0,10.5),(48.5,7.5),(42.0,-11.75),(44.5,-1.75),(39.75,-9.0),(50.25,4.75),(55.25,2.75),(36.25,-19.25),(63.0,-15.75),(33.75,-9.25),(46.75,-14.75),(50.0,5.5),(36.25,-9.25),(61.5,-5.5),(62.75,6.75),(53.0,-12.75),(36.25,-13.25),(59.25,-15.25),(64.25,-10.5),(41.75,-13.75),(58.5,-12.75),(34.5,-12.75),(59.0,17.75),(48.0,17.5),(64.0,4.75),(55.0,-5.25),(61.25,4.25),(36.25,12.75),(53.75,12.25),(40.0,-8.75),(43.25,1.75),(36.0,-9.5),(63.25,10.75),(51.0,3.75),(64.75,-0.5),(36.5,16.25),(56.25,-14.25),(48.5,11.25),(56.0,3.5),(64.0,-15.5),(64.75,13.5),(36.25,-9.5),(37.75,18.5),(54.0,14.25),(44.0,14.0),(47.5,19.75),(35.25,-6.25),(62.25,9.75),(55.25,17.0),(34.75,13.5),(52.75,-6.25),(42.25,-10.75),(35.75,-8.5),(30.75,-4.25),(55.75,-16.25),(41.75,9.0),(61.75,11.25),(64.75,15.5),(61.75,-12.25),(33.75,17.5),(40.5,-10.75),(37.25,-19.0),(36.75,-0.25),(46.0,-17.75),(58.75,17.0),(36.0,17.75),(44.0,7.5),(43.0,-13.5),(43.5,9.0),(51.25,16.75),(64.0,-18.5),(43.75,2.0),(31.5,-19.5),(38.0,0.25),(61.5,17.0),(41.75,12.5),(61.25,17.5),(46.0,6.5),(47.75,1.5),(44.5,-6.5),(50.25,18.25),(41.0,-6.0),(36.25,6.5),(37.25,9.75),(37.0,-15.5),(43.75,-10.5)]
#n=1000:
training_locations1000 = [(48.75,-12.25),(64.75,-4.25),(64.5,-0.75),(38.0,9.5),(64.25,1.75),(51.0,-3.75),(60.75,2.25),(40.5,13.5),(32.25,-16.0),(43.0,7.25),(35.75,17.75),(32.25,13.75),(53.75,-10.5),(47.0,-8.25),(44.5,4.5),(51.5,19.0),(47.5,-18.25),(47.0,-12.25),(50.25,-19.75),(62.0,16.0),(33.75,5.25),(51.0,-16.5),(49.25,-10.25),(64.25,-12.5),(45.75,1.0),(43.5,12.25),(62.25,-15.0),(37.0,-16.5),(44.75,1.25),(39.5,5.5),(52.25,-15.75),(59.25,-5.5),(62.5,9.75),(59.25,4.25),(43.25,4.0),(44.75,20.25),(32.75,-13.5),(40.0,16.25),(32.25,12.25),(39.75,0.0),(55.25,-1.5),(59.0,-5.25),(33.5,1.75),(32.25,-19.0),(35.5,-2.75),(62.25,10.75),(59.75,17.75),(56.5,-6.5),(37.75,17.0),(63.5,10.75),(47.75,-14.75),(52.5,20.0),(49.75,-9.25),(46.75,-1.5),(63.25,18.5),(50.75,17.25),(48.75,3.5),(45.75,2.75),(38.75,5.5),(62.25,-5.5),(43.5,-2.5),(37.0,9.0),(30.5,-8.25),(40.0,-2.5),(48.75,-5.25),(36.75,-0.5),(60.75,-17.0),(47.75,-12.0),(41.75,9.25),(62.5,-15.0),(36.75,1.25),(36.25,-7.5),(58.0,-8.5),(42.5,-1.0),(64.25,13.75),(45.5,20.0),(33.5,9.25),(33.75,-14.25),(37.5,-13.0),(43.25,3.5),(36.5,-5.0),(46.5,13.5),(55.25,-9.0),(42.25,-8.5),(58.0,8.75),(42.0,-5.25),(54.5,14.0),(57.75,-17.5),(40.5,2.5),(39.0,9.0),(41.5,16.5),(57.0,20.0),(35.75,5.5),(31.75,-6.0),(39.75,-14.5),(48.25,1.25),(50.5,-3.25),(32.75,7.25),(36.0,-12.25),(50.75,-11.0),(44.75,14.0),(41.25,5.0),(49.25,10.0),(48.75,-8.5),(42.5,-8.75),(50.75,-0.5),(65.0,-3.0),(44.25,-13.5),(48.75,-12.25),(35.5,9.0),(39.25,-17.25),(48.5,4.5),(59.25,-1.75),(44.25,-3.0),(42.25,-6.0),(34.75,-7.75),(34.25,6.25),(34.25,5.75),(53.5,11.75),(61.75,-15.5),(56.75,16.5),(33.5,-15.25),(62.75,-4.75),(39.75,-3.0),(54.25,2.0),(48.75,10.5),(52.25,16.5),(40.25,-9.75),(56.25,8.75),(34.25,6.75),(46.75,12.0),(63.5,2.25),(34.5,-5.75),(42.5,5.5),(40.5,-19.5),(49.25,-18.75),(56.75,4.75),(35.5,15.0),(46.75,-16.25),(59.0,-14.75),(39.0,13.0),(35.0,-6.25),(42.0,-3.25),(45.0,1.5),(44.75,0.5),(57.75,16.25),(44.25,3.75),(49.75,-7.5),(44.5,7.75),(49.75,14.5),(45.0,17.0),(33.75,13.0),(43.75,-17.25),(43.75,6.0),(64.75,1.75),(59.75,4.5),(58.5,-3.5),(48.0,-0.75),(43.5,19.0),(34.25,2.25),(33.5,-13.0),(55.0,0.0),(55.75,2.0),(63.5,-8.0),(47.0,-6.25),(38.25,8.0),(46.25,-15.5),(57.5,-2.5),(49.5,-15.25),(59.25,-12.25),(60.0,0.5),(50.25,18.75),(32.75,-20.0),(48.25,-16.75),(42.5,-2.25),(40.5,11.5),(38.25,-17.75),(36.0,-18.75),(45.0,-14.75),(39.5,2.75),(30.25,13.5),(45.75,-19.5),(57.75,-9.25),(39.75,-20.0),(40.75,8.5),(31.75,13.5),(45.0,-5.25),(54.75,-18.25),(62.25,1.5),(42.25,-2.5),(36.0,-11.0),(38.25,11.5),(53.75,-18.25),(52.75,-1.5),(37.75,-9.5),(62.5,-13.25),(39.25,-16.5),(41.0,-16.0),(58.75,-19.0),(59.5,-14.25),(52.75,-7.25),(59.0,-9.0),(52.25,-1.0),(49.25,14.75),(58.5,-5.0),(31.25,-11.25),(46.25,-20.0),(53.25,4.0),(56.75,-11.25),(32.75,-18.25),(54.5,-12.0),(62.75,12.75),(51.75,-7.5),(47.75,-13.0),(44.0,8.5),(40.75,-5.25),(49.25,-16.25),(30.75,3.5),(31.5,-16.75),(56.5,12.75),(40.5,-14.25),(30.5,-15.0),(62.5,4.0),(42.25,-18.75),(60.25,-6.25),(31.5,-16.25),(46.5,-1.25),(32.5,4.75),(50.75,-5.5),(57.25,-7.5),(36.0,-18.75),(52.75,6.0),(55.5,0.0),(42.25,-5.75),(41.25,-18.0),(60.0,7.5),(42.25,7.0),(51.25,-3.0),(41.75,-0.0),(55.0,3.5),(55.75,-5.5),(31.5,-12.75),(40.75,-6.0),(31.75,-19.0),(60.25,-11.75),(34.75,-0.75),(63.25,-11.5),(38.75,0.25),(36.75,2.0),(43.0,7.0),(38.0,5.25),(38.5,-4.75),(52.0,-10.75),(57.0,-16.25),(55.0,-16.0),(36.75,-14.0),(61.0,-14.75),(46.75,-9.0),(30.25,10.75),(60.0,13.5),(62.5,-16.25),(32.5,16.5),(46.25,-9.25),(30.75,5.5),(41.75,6.25),(59.5,-2.75),(62.0,-0.5),(41.0,9.0),(42.75,5.5),(31.25,-5.5),(45.25,15.75),(44.75,-9.75),(37.75,-6.25),(58.25,-3.0),(57.5,-6.25),(43.5,19.25),(47.0,0.0),(55.25,-13.5),(46.25,18.0),(39.25,4.25),(47.25,-3.5),(30.5,-18.25),(32.75,7.5),(35.5,-17.25),(34.75,2.5),(51.75,-17.5),(46.5,-2.0),(31.25,-18.5),(44.5,-19.0),(52.0,-19.5),(53.5,-15.5),(46.75,6.25),(40.5,17.75),(58.75,-9.25),(49.0,-17.25),(62.5,3.25),(55.5,15.75),(63.5,14.25),(61.75,-19.25),(32.0,17.75),(45.75,-15.5),(32.0,-2.25),(61.0,-15.0),(57.0,-5.0),(47.0,-6.0),(48.25,-11.75),(47.75,-4.25),(62.5,-10.25),(58.25,14.75),(62.0,-11.25),(44.0,0.75),(39.0,-10.5),(55.25,-14.25),(46.25,0.25),(46.0,1.5),(41.0,19.25),(48.0,14.25),(64.0,19.0),(45.25,11.75),(54.5,-9.25),(58.5,6.5),(54.0,-13.5),(48.25,-14.75),(49.25,-15.5),(52.0,-17.0),(39.25,-5.75),(44.0,17.25),(58.75,10.75),(56.75,0.75),(57.75,-16.75),(38.5,18.75),(41.5,-12.75),(54.5,-4.0),(43.0,10.5),(31.25,-3.25),(37.75,13.25),(38.75,8.0),(31.0,3.25),(43.75,-12.0),(42.5,8.5),(45.25,2.0),(37.5,-16.75),(61.25,-2.25),(61.25,-10.25),(43.75,1.0),(52.0,5.5),(31.25,-6.75),(59.25,-14.5),(64.75,5.0),(53.75,-19.25),(62.75,-14.25),(49.75,-17.75),(60.75,-2.75),(64.25,11.75),(31.5,11.75),(63.5,-15.0),(53.25,-19.25),(56.0,5.75),(38.5,-10.0),(50.0,-16.5),(59.25,-1.0),(32.25,-3.5),(50.0,12.5),(47.0,-1.5),(58.25,0.75),(51.75,-13.75),(52.25,-10.5),(33.5,15.0),(35.25,20.25),(32.0,-4.0),(61.0,6.5),(49.5,14.25),(61.0,-16.5),(36.25,13.5),(53.0,-10.5),(64.25,14.5),(62.0,-15.0),(48.5,-9.75),(47.0,-16.25),(46.5,4.25),(57.0,-7.75),(52.75,1.25),(61.5,-13.25),(32.0,2.75),(64.0,4.5),(51.75,2.0),(54.75,-18.25),(45.75,18.5),(46.5,-0.25),(46.25,14.75),(35.5,-18.5),(39.75,-7.5),(37.25,15.5),(59.0,-14.5),(50.0,11.0),(38.0,9.0),(47.75,8.25),(50.75,10.25),(38.0,-9.0),(39.0,1.25),(47.5,-16.0),(45.5,0.75),(32.5,19.5),(58.75,-14.25),(31.25,8.25),(56.5,-8.75),(57.0,-10.0),(52.5,-18.75),(37.0,-12.5),(51.0,12.25),(50.5,4.25),(58.5,7.0),(33.75,-16.5),(41.0,-5.75),(48.25,18.5),(34.75,-5.75),(61.75,2.5),(37.5,6.0),(56.75,9.0),(55.5,6.75),(61.5,-5.75),(33.25,-8.0),(52.25,-19.0),(49.0,-7.0),(63.25,13.25),(36.0,8.75),(33.5,-2.0),(49.5,18.25),(32.25,0.25),(64.5,-19.75),(61.5,-19.0),(63.0,5.5),(30.75,-6.25),(42.75,12.5),(54.5,-12.5),(39.5,2.5),(51.25,18.0),(58.0,12.25),(39.75,4.5),(56.25,-14.75),(40.75,-4.0),(32.5,1.75),(60.0,7.75),(53.5,4.75),(54.5,-11.5),(49.0,8.75),(34.0,-9.5),(58.25,19.25),(35.5,-11.75),(62.0,18.5),(52.0,1.75),(38.75,-16.0),(62.25,5.25),(61.75,15.25),(50.25,-16.0),(60.25,2.75),(55.0,-0.25),(44.5,2.5),(42.25,-17.25),(51.75,10.0),(45.5,14.5),(58.0,14.25),(43.75,8.5),(33.0,9.25),(31.5,-9.75),(64.0,16.5),(32.5,3.5),(39.5,-7.0),(64.25,10.25),(46.0,12.0),(47.25,12.25),(33.25,17.25),(59.75,-9.25),(44.0,-5.5),(43.75,-0.25),(43.75,-6.75),(54.5,6.5),(44.25,10.75),(37.25,1.0),(56.75,19.0),(39.25,0.25),(41.75,-16.75),(49.25,-10.0),(52.5,-14.5),(33.5,-9.25),(59.0,-20.0),(31.75,-0.0),(63.5,2.75),(48.5,-13.25),(57.5,17.5),(63.75,-15.5),(32.25,15.0),(52.75,-17.0),(31.75,12.0),(58.0,-19.0),(46.0,-14.75),(62.25,16.25),(42.25,-13.0),(61.0,9.75),(43.75,14.25),(49.75,-14.0),(57.75,-0.5),(56.25,3.5),(33.5,13.0),(62.0,12.75),(64.5,-13.25),(36.0,14.25),(61.25,17.5),(55.5,4.75),(54.25,-0.75),(48.0,0.5),(37.25,-19.0),(48.0,12.0),(49.5,-11.0),(61.5,18.0),(61.5,-1.25),(38.25,16.75),(55.5,14.25),(64.75,1.75),(46.25,17.5),(56.5,-11.0),(34.0,-0.75),(61.5,13.5),(33.0,10.0),(31.75,17.25),(39.5,-6.25),(44.25,17.0),(41.0,-6.25),(63.0,-14.25),(40.0,17.25),(42.25,-13.75),(41.75,-5.75),(62.75,-12.75),(64.5,-14.5),(63.0,-9.75),(62.75,4.5),(62.5,10.25),(35.75,-0.5),(31.0,2.75),(45.5,-8.0),(57.75,-8.0),(40.0,18.5),(55.5,1.75),(45.25,19.5),(54.0,1.75),(64.75,-18.5),(33.5,-18.0),(62.25,17.75),(31.0,14.5),(57.25,-9.25),(47.75,15.5),(46.75,-4.25),(36.75,-5.25),(41.25,7.25),(42.0,3.5),(38.75,-5.75),(64.5,11.5),(62.25,-18.5),(51.25,0.75),(48.0,7.5),(32.5,-10.0),(39.25,-4.0),(40.25,2.75),(62.75,-7.5),(58.75,8.5),(62.75,-5.75),(59.5,6.5),(42.5,-14.75),(49.5,-16.75),(31.25,-11.5),(62.75,5.75),(60.0,-10.75),(57.0,-5.25),(52.25,-17.25),(60.0,-8.5),(32.25,-10.5),(39.5,-6.0),(47.0,0.0),(30.25,9.75),(30.75,10.75),(61.5,15.75),(32.75,8.5),(49.25,6.5),(47.25,-0.5),(61.75,-14.75),(49.5,14.0),(62.0,-20.0),(58.75,14.25),(51.5,-19.25),(63.25,18.5),(35.25,-1.25),(59.5,19.5),(32.25,-8.0),(63.75,-6.75),(60.75,-7.25),(53.75,-7.25),(37.5,-12.5),(42.0,3.25),(43.0,20.25),(59.75,-7.75),(35.75,12.5),(65.0,19.25),(56.0,17.25),(63.25,14.75),(48.5,-0.5),(51.25,-10.5),(42.75,3.0),(39.5,9.0),(40.0,-8.0),(51.0,5.75),(62.75,-15.75),(49.25,14.25),(31.25,-17.0),(34.0,-13.5),(39.5,16.5),(49.25,15.5),(64.0,12.25),(41.0,-8.25),(43.5,11.5),(47.25,-18.0),(39.5,8.25),(44.5,18.75),(45.0,1.5),(42.5,-2.0),(64.5,-5.0),(53.75,15.25),(57.25,6.0),(58.5,-3.25),(43.75,19.0),(56.5,9.25),(54.25,12.0),(33.5,6.0),(40.0,4.25),(38.5,-3.0),(51.0,-18.25),(40.25,11.75),(55.0,0.0),(63.25,6.0),(32.0,-17.0),(47.0,-16.0),(33.5,3.0),(36.5,13.5),(51.25,6.25),(34.5,11.25),(42.5,-17.25),(61.25,-19.0),(49.25,-18.5),(44.25,8.25),(32.5,-1.5),(46.75,-8.5),(48.75,13.25),(49.25,0.75),(46.5,-5.25),(64.0,6.0),(47.5,9.75),(43.75,-3.5),(38.5,8.5),(38.0,-2.75),(37.75,-17.75),(40.5,17.0),(49.25,6.75),(47.75,14.75),(65.0,13.5),(31.5,-4.25),(36.0,-15.5),(62.5,12.25),(32.25,17.25),(63.25,-12.75),(63.25,13.25),(38.75,-17.0),(63.75,7.5),(57.0,0.25),(54.0,-10.25),(49.0,19.25),(50.0,13.5),(55.75,10.0),(32.75,5.5),(35.75,17.0),(49.5,20.0),(58.25,-14.25),(30.5,-10.5),(39.25,-3.5),(47.5,-3.25),(55.25,11.75),(57.5,6.25),(31.0,-4.75),(54.0,-11.0),(64.75,11.25),(52.25,9.5),(42.5,-14.5),(32.75,-13.75),(63.5,-15.5),(56.75,10.75),(55.5,15.0),(34.75,2.0),(62.75,14.0),(45.5,18.25),(61.75,-14.0),(45.25,-19.75),(57.25,-15.5),(60.75,8.25),(36.5,-5.5),(50.75,9.75),(51.0,-12.0),(33.5,-13.25),(50.5,13.0),(46.0,1.5),(38.25,1.25),(55.75,12.0),(50.0,-1.5),(63.25,-19.25),(57.25,3.0),(63.0,-4.0),(43.5,-7.0),(42.0,-5.25),(33.25,-7.25),(31.5,-10.25),(50.75,-1.5),(63.25,-15.5),(35.25,4.5),(50.5,14.25),(35.75,20.0),(32.25,5.0),(61.5,-7.25),(49.75,17.0),(33.25,-0.75),(52.75,-4.5),(36.75,4.25),(60.0,-12.25),(37.75,16.5),(36.75,-2.0),(47.75,19.25),(44.5,-4.75),(41.5,-6.75),(50.25,-13.5),(41.5,-11.25),(36.25,11.25),(45.75,2.75),(44.5,10.0),(43.0,20.0),(41.5,10.5),(31.0,9.5),(54.5,-4.25),(40.75,-1.5),(56.75,19.5),(30.5,-13.0),(56.0,14.25),(33.75,5.5),(39.75,-17.5),(38.5,11.0),(41.0,13.5),(44.25,9.75),(31.75,-1.5),(36.0,8.25),(43.5,-11.0),(59.5,9.75),(61.0,-0.25),(60.75,-15.25),(60.0,-13.25),(41.0,-18.5),(37.75,-8.25),(58.75,-8.0),(36.5,2.25),(45.5,-12.5),(58.5,-14.5),(51.25,-0.75),(62.5,-4.75),(36.5,-15.75),(43.75,2.25),(59.25,8.5),(51.25,-3.0),(62.25,13.0),(38.75,-0.75),(45.0,-20.0),(51.0,11.25),(36.5,18.0),(59.5,6.5),(32.5,-5.5),(62.75,11.75),(55.75,9.5),(42.0,15.5),(61.0,4.75),(32.75,19.25),(44.0,1.0),(61.5,-0.75),(50.0,-17.0),(33.25,4.5),(63.25,14.25),(44.75,-13.5),(49.5,7.75),(43.0,9.5),(47.25,14.0),(32.75,-5.25),(64.75,6.25),(34.5,-0.25),(57.0,3.5),(61.25,0.75),(34.25,-5.75),(34.25,-14.75),(41.0,-13.25),(38.5,14.0),(45.0,-10.25),(62.75,-5.25),(34.75,5.5),(58.25,-13.75),(41.5,13.75),(36.75,-18.25),(55.75,-19.75),(45.75,12.5),(50.25,6.0),(45.0,-4.75),(45.5,-19.5),(37.25,-7.0),(45.25,4.5),(51.5,-18.0),(33.0,-10.75),(42.25,14.5),(45.25,-12.0),(59.75,-13.75),(41.25,5.5),(62.5,9.75),(63.0,-19.0),(43.25,-5.5),(52.25,-11.5),(59.5,-14.75),(49.25,-5.5),(46.25,19.0),(61.25,10.0),(62.75,2.25),(56.25,-8.5),(30.5,-10.5),(31.0,-17.75),(52.25,-1.5),(39.75,-2.75),(37.75,2.75),(44.75,18.0),(64.75,15.5),(34.0,-14.0),(41.75,-5.5),(31.5,-5.5),(46.75,17.5),(32.75,7.5),(42.25,-4.0),(52.75,-4.0),(60.0,6.5),(49.0,-0.25),(49.0,6.25),(31.0,4.5),(64.0,-3.25),(45.0,3.25),(47.5,17.5),(55.5,-19.0),(39.25,3.0),(30.25,-17.5),(44.5,14.75),(36.0,-8.25),(52.75,14.75),(44.25,18.25),(44.25,-4.0),(52.25,7.5),(60.5,10.5),(41.0,2.5),(43.5,14.75),(47.0,9.25),(36.0,-5.75),(30.25,5.5),(47.5,7.25),(40.0,-14.75),(63.0,18.5),(55.75,-14.25),(39.0,-12.5),(38.5,-8.25),(61.75,4.0),(30.5,-5.75),(33.0,-2.75),(48.75,-6.75),(47.75,-6.0),(57.0,-8.5),(40.75,-15.0),(47.5,10.75),(59.5,-4.25),(46.0,14.0),(47.25,-12.75),(30.25,1.0),(32.25,11.5),(41.75,17.25),(63.0,6.5),(60.75,-10.0),(51.0,11.5),(60.25,5.25),(37.0,-2.0),(34.0,-8.5),(45.25,4.0),(60.75,11.25),(62.75,-13.25),(60.0,-2.0),(51.5,-2.25),(49.0,1.5),(37.75,19.25),(42.5,-8.5),(57.5,16.25),(46.75,-1.5),(52.0,-15.0),(57.5,8.0),(35.5,-18.0),(55.25,-6.25),(60.5,13.5),(45.25,2.5),(32.5,7.75),(41.25,-15.25),(41.5,-10.25),(33.0,2.25),(42.5,10.25),(32.25,-7.25),(36.75,-4.5),(61.75,-3.75),(51.5,-15.25),(32.75,-11.0),(38.5,8.75),(33.75,2.0),(46.5,-6.5),(37.25,13.25),(54.75,-3.25),(37.25,-0.5),(60.0,12.75),(52.25,-16.75),(41.25,11.0),(50.5,-11.75),(49.5,-4.5),(64.5,-12.0),(38.0,9.5),(59.25,1.5),(57.0,-5.75),(32.75,-6.5),(55.5,-2.5),(64.25,16.75),(60.25,-17.0),(59.25,-13.5),(63.25,8.25),(57.5,9.75),(61.75,-5.0),(43.0,-0.5),(63.75,-11.75),(35.25,8.5),(60.0,18.25),(43.75,10.75),(38.25,-16.25),(61.75,19.5),(61.0,-6.5),(39.5,12.5),(52.25,-19.75),(34.5,-16.0),(31.0,-16.0),(41.75,-0.25),(39.75,19.25),(32.75,-7.25),(32.75,6.75),(42.5,-4.5),(61.75,-16.75),(61.75,3.0),(47.75,16.0),(43.5,15.75),(44.0,-5.5),(53.0,0.25),(57.25,-13.75),(38.25,-19.25),(45.0,-15.25),(49.75,16.25),(32.0,14.5),(47.5,19.75),(51.25,8.0),(60.75,-7.25),(39.5,-5.25),(64.25,1.75),(37.25,8.75),(46.0,11.75),(42.0,15.5),(32.25,3.75),(65.0,0.25),(53.75,-8.75),(31.25,-16.5),(40.5,-7.5),(63.5,17.75),(43.25,19.5),(39.0,17.0),(41.75,-17.75),(41.5,-3.5),(64.25,1.75),(45.75,3.75),(58.25,-15.0),(52.0,-2.5),(33.25,14.0),(62.0,-0.25),(63.0,16.5),(34.25,-13.25),(51.0,-16.75),(32.5,10.75),(56.75,13.5),(58.5,-11.5),(59.5,-16.5)]

# round to 1x1 data
training_locations10_1x1 = np.round(training_locations10)
training_locations100_1x1 = np.round(training_locations100)
training_locations1000_1x1 = np.round(training_locations1000)


# Special_locations:
# TODO include special locations then make list