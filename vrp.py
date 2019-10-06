from pprint import pprint
import time as tm
import gmpy2
import json
import requests
import sys
import itertools
import random
import numpy as np

min_cost = [None,float("inf")]

def prng(j, g = 0, z = None):
    zr = []
    rnd = [{"a":354623, "z":179875, "m":236481},
            {"a":11234, "z":65489, "m":236417},
            {"a":76982, "z":17456, "m":33651},
            {"a":152717, "z":135623, "m":210321},
            {"a":197331, "z":172361, "m":254129},
            {"a":48271, "z":172361, "m":2**31-1}, #Lehmer CG
            {"a":1071064 , "z":[135623,172361], "m":2**31-19}, #MRG
            {"a":6364136223846793005, "z":172361, "m":2**64}, #LCG
            {"a":197331, "z":172361, "m":2**31-1}, #ICG
            {"a":197331, "z":[172361, None], "m":2**48-59}] #EICG
    if g in [0,1,2,3,4]:
        z = z if z != None else rnd[g]["z"]
        a, m = rnd[g]["a"], rnd[g]["m"]
        for n in range(j):
            z = a*z%m
            zr += [z%(j-n)]
    elif g == 5: # Lehmer
        z = z if z != None else rnd[g]["z"]
        a, m = rnd[g]["a"], rnd[g]["m"]
        for n in range(j):
            z = (a*z)%m
            zr += [z%(j-n)]
    elif g == 6: # MRG
        z = z if z != None else rnd[g]["z"]
        a, m = rnd[g]["a"], rnd[g]["m"]
        for n in range(j):
            z[0], z[1] = z[1], (a*z[1]+2113664*z[0])%m
            zr += [z[1]%(j-n)]
    elif g == 7: #LCG
        z = z if z != None else rnd[g]["z"]
        a, m = rnd[g]["a"], rnd[g]["m"]
        for n in range(j):
            z = (a*z+1)%m
            zr += [z%(j-n)]
    elif g == 8: #ICG
        z = z if z != None else rnd[g]["z"]
        a, m = rnd[g]["a"], rnd[g]["m"]
        for n in range(j):
            z = (gmpy2.invert(z,m)+1)%m
            zr += [z%(j-n)]
    elif g == 9: #EICG
        z = z if z != None else rnd[g]["z"]
        a, m = rnd[g]["a"], rnd[g]["m"]
        for n in range(j):
            z[-1] = gmpy2.invert(n+1+z[0],m)
            zr += [z[n]%(j-n)]
    return zr, z


# Monte Carlo VRP solver algorithm
def mc(vehicles, jobs, M, epochs=100000, p=1000):
    j, z = len(jobs), None
    routes = []
    now = tm.time()
    for n in range(epochs):
        
        if n%10000 == 0:
            print("epoch: ", n, "time elapsed: ", tm.time()-now)

        zr, z = prng(j, g=4, z=z)

        h, cost = 0, 0
        route = {} # Initialize route list for this partition
        jobs_idx = [i for i in range(j)] # Jobs' index array
        for vehicle in vehicles:
            v_name = vehicle['id'] # Construct current vehicle name
            route[v_name] = [] 
            delivery, i = 0, 0
            time = vehicle["time_window"][0]
            i0, i1 = None, None
            while time < vehicle["time_window"][1]:

                job = jobs[jobs_idx[zr[h]]]
                job_keys = job.keys()

                # Check the eligibility of the vehicle's skills for the job
                try:
                    if not all(elem in vehicle["skills"] for elem in job["skills"]):
                        break
                except KeyError: pass

                try:
                    # First-element-only cumulative delivery-amount
                    delivery += job["amount"][0] 
                    # Check for the cumulative delivery-amount 
                    if delivery > vehicle["capacity"][0]:
                        break
                except KeyError: pass

                # Check arrival time
                if "time_window" in job_keys and (time < job["time_window"][0] or time > job["time_window"][1]):
                    break
                
                i1 = job["location_index"]
                if i == 0 and "start_index" in vehicle.keys():
                    i0 = vehicle["start_index"]
                    cost += M[i0][i1]
                    time += M[i0][i1]
                    i0 = i1
                # elif i == val - 1 and "end_index" in vehicle.keys():
                #     i0 = i1
                #     i1 = vehicle["end_index"]
                #     cost[-1] += M[i0][i1]
                #     time += M[i0][i1]                        
                else:
                    cost += M[i0][i1]
                    time += M[i0][i1]
                    i0 = i1
                
                time += job["service"]

                # Assign the node to current vehicle as next job
                route[v_name] +=  [job["id"]]
                del jobs_idx[zr[h]]
                h += 1
                i += 1
            else: continue
            break
        else:
            routes += [route, cost]
            # cost = sum(cost)
            if cost < min_cost[1]:
                min_cost[0] = list(route.values())
                min_cost[1] = cost
                print("min cost", cost)
    return routes


def cost_matrix(vehicles, jobs):
    req = "http://localhost:5000/table/v1/car/"
    points = []
    for vehicle in vehicles:
        if "start" in vehicle:
            try:
                idx_start = points.index(vehicle["start"])
                vehicle["start_index"] = idx_start
            except:
                points += [vehicle["start"]]
                vehicle["start_index"] = len(points) - 1
                req += str(vehicle["start"][0]) + "," + str(vehicle["start"][1]) + ";"
        if "end" in vehicle:       
            try:
                idx_end = points.index(vehicle["end"])
                vehicle["end_index"] = idx_end
            except:
                points += [vehicle["end"]]
                vehicle["end_index"] = len(points) - 1
                req += str(vehicle["end"][0]) + "," + str(vehicle["end"][1]) + ";"

    for job in jobs:
        try:
            idx_job = points.index(job["location"])
            job["location_index"] = idx_job
        except:
            points += [job["location"]]
            job["location_index"] = len(points) - 1
            req += str(job["location"][0]) + "," + str(job["location"][1]) + ";"
    req = req.rstrip(";")
    res = requests.get(req)
    if res.status_code != 200:
        sys.exit('Error while requesting cost matrix!')

    M = res.json()["durations"]
    return vehicles, jobs, M # In python, lists are immutable which means they passed via 'pass by reference'
                            # so that, in fact, we don't need to return vehicles and jobs


def read_data(path, process = False):
    with open(path) as data:
        data = json.load(data)
        if process:
            vehicles, jobs, M = cost_matrix(data["vehicles"], data["jobs"])
            return vehicles, jobs, M
        else:
            return data["vehicles"], data["jobs"], data["M"]

def persist_data(path, **data):
    with open(path, 'w') as json_file:
        output = {"vehicles": data["vehicles"], "jobs": data["jobs"], "M": data["M"]}
        json.dump(output, json_file)


################################################################

vehicles, jobs, M = read_data("input_thu.json")
# persist_data("input_thu.json", vehicles=vehicles, jobs=jobs, M=M, partitions=partitions)

epochs = 5*10**6
p = 10**5
start = tm.time()

mc(vehicles=vehicles, jobs=jobs, M=M, epochs=epochs, p=p)

print("Time elapsed: " + str(tm.time() - start))
print("Min cost: ", min_cost)

#45s 1433 cost