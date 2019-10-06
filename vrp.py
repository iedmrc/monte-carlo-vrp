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

def partitions(j, v):
    i = 0
    f = .35
    while v != 0:
        if v == 1:
            v -= 1
            yield i, j
        else:
            r = int(abs(np.random.normal(j/v, (j/v)*f, 1)[0]))
            j = j - r
            v -= 1
            f = round(f * .9, 4)
            yield i, r
        i += 1

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
    j, v = len(jobs), len(vehicles)
    routes = []
    maxl, z = 0, None
    now = tm.time()
    for n in range(epochs):

        print("epoch: ", n, "time elapsed: ", tm.time()-now)

        zr, z = prng(j, g=8, z=z)

        # Loop over each partition
        for _ in range(p):
            route = {} # Initialize route list for this partition
            h = 0
            cost = []
            jobs_idx = [i for i in range(j)] # Jobs' index array
            for idx, val in partitions(j,v):
                v_name = vehicles[idx]['id'] # Construct current vehicle name
                route[v_name] = [] 
                delivery = 0
                cost += [0]
                time = vehicles[idx]["time_window"][0]
                i0, i1 = None, None
                for _ in range(val):

                    # job = next((item for item in jobs if item["id"] == jobs_idx[zr[h]]))
                    job = jobs[jobs_idx[zr[h]]]
                    job_keys = job.keys()
                    # Check the eligibility of the vehicle's skills for the job
                    try:
                        if not all(elem in vehicles[idx]["skills"] for elem in job["skills"]):
                            break
                    except KeyError: pass

                    try:
                        # First-element-only cumulative delivery-amount
                        delivery += job["amount"][0] 
                        # Check for the cumulative delivery-amount 
                        if delivery > vehicles[idx]["capacity"][0]:
                            break
                    except KeyError: pass

                    # Check arrival time
                    if "time_window" in job_keys and (time < job["time_window"][0] or time > job["time_window"][1]):
                        break
                    
                    i1 = job["location_index"]
                    if _ == 0 and "start_index" in vehicles[idx].keys():
                        i0 = vehicles[idx]["start_index"]
                        cost[-1] += M[i0][i1]
                        time += M[i0][i1]
                        i0 = i1
                    elif _ == val - 1 and "end_index" in vehicles[idx].keys():
                        i0 = i1
                        i1 = vehicles[idx]["end_index"]
                        cost[-1] += M[i0][i1]
                        time += M[i0][i1]                        
                    else:
                        cost[-1] += M[i0][i1]
                        time += M[i0][i1]
                        i0 = i1
                    
                    time += job["service"]
                    
                    # Check working hours
                    if time > vehicles[idx]["time_window"][1]:
                        break

                    # Assign the node to current vehicle as next job
                    route[v_name] +=  [job["id"]]
                    del jobs_idx[zr[h]]
                    h += 1
                else: continue
                l = list(itertools.chain.from_iterable(route.values()))
                if len(l) > maxl:
                    print(route, len(l))
                    print("---------------------------------------------")
                    maxl = len(l)
                
                break
            else:
                routes += [route]
                cost = sum(cost)
                if cost < min_cost[1]:
                    min_cost[0] = list(route.values())
                    min_cost[1] = cost
                    print(min_cost)

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

epochs = 5*10**3
p = 10**5
start = tm.time()

mc(vehicles=vehicles, jobs=jobs, M=M, epochs=epochs, p=p)

print("Time elapsed: " + str(tm.time() - start))
print("Min cost: ", min_cost)

#45s 1433 cost