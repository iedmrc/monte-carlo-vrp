from pprint import pprint
import time
import gmpy2
import json
import requests
import sys
import itertools
import random

min_cost = [None,float("inf")]

# Cost Function
def cost(route, m, vehicles, jobs):
    cost = 0
    for _id, rt in route.items():
        for i in range(len(rt)-1):
            i0 = next((val["location_index"] for (idx,val) in enumerate(jobs) if val["id"] == rt[i]))
            i1 = next((val["location_index"] for (idx,val) in enumerate(jobs) if val["id"] == rt[i+1]))
            cost += m[i0][i1]
            if i == 0:
                try:
                    start_index = next((val["start_index"] for (idx,val) in enumerate(vehicles) if val["id"] == _id))
                    cost += m[start_index][i0] # Start index cost
                except KeyError: pass
            if i == len(rt)-2:
                try:
                    end_index = next((val["end_index"] for (idx,val) in enumerate(vehicles) if val["id"] == _id))
                    cost += m[i1][end_index] # End index cost
                except KeyError: pass
    if cost < min_cost[1]:
        min_cost[0] = list(route.values())
        min_cost[1] = cost
        print(min_cost)
    return cost


# Monte Carlo VRP solver algorithm
def mc(rnd, vehicles, jobs, M, partitions, epochs=1000):
    # Random number parameters
    a, z, m, j = rnd["a"], rnd["z"], rnd["m"], len(jobs)
    routes = []
    for n in range(1,epochs):
        # Create a new random number and append to the `z` array
        z += [(a*z[n]+2113664*z[n-1])%m] #MRG
        # z += [(a*z[n])%m] #Lehmer CG
        #z += [(a*z[n]+1)%m] #LCG
        # z += [(gmpy2.invert(z[n],m)+1)%m] #ICG
        #z += [gmpy2.invert(n+1+z[0],m)] #EICG
        # z += [a*z[n]%m]
        # The modulos array of the random number

        #zr = [z[n+1]%(j-i) for i in range(j)]

        if not ((n+1)%j == 0):
            continue
        zr = [z[n+1-i]%(j-i) for i in range(j)]

        # Loop over each partition
        for partition in partitions:
            route = {} # Initialize route list for this partition
            h = 0
            cost = []
            jobs_idx = [i for i in range(j)] # Jobs' index array
            for idx, val in enumerate(partition):
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
                break
            else:
                routes += [route, z[-1]]
                cost = sum(cost)
                if cost < min_cost[1]:
                    min_cost[0] = list(route.values())
                    min_cost[1] = cost
                    print(min_cost)
                # cost(route, M, vehicles, jobs)
                # print(route,cost(route,M,jobs))

    # pprint(routes)
    # print(rnd,'\n')
    return routes


def cost_matrix(vehicles, jobs):
    #32.853138888,39.919305555
    #"http://localhost:5000/table/v1/car/32.862167,39.931500;32.782417,39.895944;32.823111,39.942472;"
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
    return M, vehicles, jobs # In python, lists are immutable which means they passed via 'pass by reference'
                            # so that, in fact, we don't need to return vehicles and jobs

def accel_asc(n, p):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            D = max(a[:k + 2]) - min(a[:k + 2])
            if len(a[:k + 2]) == p and 1 not in a[:k + 2] and D <= n//p + 1:
                yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        D = max(a[:k + 1]) - min(a[:k + 1])
        if len(a[:k + 1]) == p and 1 not in a[:k + 1] and D <= n//p + 1:
            yield a[:k + 1]

def read_data(path, process = False):
    with open(path) as data:
        data = json.load(data)
        if process:
            M, vehicles, jobs = cost_matrix(data["vehicles"], data["jobs"])
            v, j = len(vehicles), len(jobs)
            # partitions = [*itertools.chain.from_iterable(set(itertools.permutations(p)) for p in accel_asc(j, v))]
            partitions = [j//v for i in range(v)]
            r = j - ((j//v)*len(partitions))
            for i in range(r):
                partitions[i] += 1
            partitions = [partitions]
            for _ in range(len(partitions[-1])):
                cp = partitions[-1].copy()
                random.shuffle(cp)
                partitions += [cp]
            return vehicles, jobs, M, partitions
        else:
            return data["vehicles"], data["jobs"], data["M"], data["partitions"]

def persist_data(path, **data):
    with open(path, 'w') as json_file:
        output = {"vehicles": data["vehicles"], "jobs": data["jobs"], "M": data["M"], "partitions": data["partitions"]}
        json.dump(output, json_file)

    




################################################################

# Define parameters
rnd_params = [
    #{"a":354623, "z":[179875], "m":236481}, #[[[5, 48, 91], [12, 33, 24], [34, 8, 13, 56]], 12167.3000000000]
    #{"a":11234, "z":[65489], "m":236417}, #[[[48, 91, 8], [5, 24, 34, 33], [13, 56, 12]], 11662.9000000000]
    #{"a":76982, "z":[17456], "m":33651}, #[[[5, 8, 91, 48], [33, 56, 24], [13, 12, 34]], 9752.80000000000]
    #{"a":152717, "z":[135623], "m":210321}, #[[[5, 8, 91, 48], [33, 24], [56, 13, 12, 34]], 9030.10000000000]
    # {"a":197331, "z":[172361], "m":254129} #[[[5, 8, 91, 48], [24, 33, 34], [56, 12, 13]], 8305.00000000000]
    # {"a":48271, "z":[172361], "m":2**31-1} #Lehmer CG [[[5, 8, 91, 48], [56, 33, 34, 24], [12, 13]], 8498.30000000000]
    {"a":1071064 , "z":[135623,172361], "m":2**31-19} #MRG [[[13, 8, 91, 48], [5, 24, 34, 33], [56, 12]], 8032.70000000000]
    #{"a":6364136223846793005, "z":[172361], "m":2**64} #LCG [[[13, 8, 91, 48], [33, 34, 24, 5], [56, 12]], 9062.80000000000]
    # {"a":197331, "z":[172361], "m":2**31-1} #ICG [[[5, 8, 91, 48], [33, 34, 24], [56, 13, 12]], 8258.50000000000]
    #{"a":197331, "z":[172361], "m":2**48-59} #EICG [[[8, 91, 48], [24, 5, 34, 33], [56, 13, 12]], 8898.10000000000]



    ]


vehicles, jobs, M, partitions = read_data("input.json", process=True)
# persist_data("input.json", vehicles=vehicles, jobs=jobs, M=M, partitions=partitions)
# partitions = [[3,3,4],[3,4,3],[4,3,3],[3,2,5],[3,5,2],[5,2,3],[5,3,2],[2,3,5],[2,5,3],[1,1,8],
#             [1,8,1],[8,1,1],[2,2,6],[2,6,2],[6,2,2],[4,4,2],[4,2,4],[2,4,4],[1,2,7],[1,7,2],
#             [7,2,1],[7,1,2],[2,1,7],[2,7,1],[1,3,6],[1,6,3],[6,3,1],[6,1,3],[3,1,6],[3,6,1],
#             [1,4,5],[1,5,4],[5,4,1],[5,1,4],[4,1,5],[4,5,1]]
epochs = 1000000
print(partitions)
start = time.time()
for rnd in rnd_params:
    mc(rnd, vehicles=vehicles, jobs=jobs, M=M, partitions=partitions, epochs=epochs)

print("Time elapsed: " + str(time.time() - start))
print("Min cost: ", min_cost)
