# this class contains all the functions that handle the sattlite data

import datetime

import math

#import draw_graphs
import visibilityGraph
import os
import sys
import time
from scipy import fft
import re
import numpy as np
import networkx as nx
import pylab as plt
import visibilityGraph_for_AC
import distance_functions
from networkx.utils import open_file

import pickle

#creates lat_lon and distance file- pickle,and  distances csv file
#q: number of times, dt: delta
falcons=65
seperations=1
distances={}
t_sat_lat_lon= []

def createFilesFromTLe(times, dt, distance_csv_file=False, lat_lon__csv_file=False, VG_files=False):

    from datetime import datetime, timedelta
    now=datetime.today()
    f_name = "distances_" + str(times) + "_" + str(dt)
    f_lat_lon_name="lat_lon"+ str(times) + "_" + str(dt)
    t_noeds=[]
    t_names=[]

    import urllib.request

    GPS_list = 'http://www.celestrak.com/NORAD/elements/gps-ops.txt'
    StarLink_list = 'http://www.celestrak.com/NORAD/elements/starlink.txt'
    GPS2_list = 'http://www.tle.info/data/gps-ops.txt'
    GLONASS_list = 'http://www.celestrak.com/NORAD/elements/glo-ops.txt'
    GLONASS2_list = 'http://www.tle.info/data/glo-ops.txt'

    import ephem
    observer = ephem.Observer()
    #dictionary of all distances pairs


    for q in range(0,times):
        sat_lat, sat_lon, sat_name = [], [],[]
        nodes = []
        sats=[]

        lastHourDateTime = now - timedelta(minutes=dt*q)

        observer.date = lastHourDateTime
        observer.lat=0
        observer.lon=0
        counter=0
        with urllib.request.urlopen(StarLink_list) as url:

                tles = url.readlines()
                #print(tles)
                tles = [item.strip() for item in tles]
                tles = [(tles[i],tles[i+1],tles[i+2]) for i in range(0,len(tles)-2,3)]

                s=""
                c=0
                for tle in tles:

                        try:

                            sat = ephem.readtle(tle[0].decode("utf-8") , tle[1].decode("utf-8") , tle[2].decode("utf-8") )
                            sats.append(sat)
                            rt, ra, tt, ta, st, sa = observer.next_pass(sat)

                            if rt is not None and st is not None:
                                sat.compute(observer)

                                text = tle[0].decode("utf-8")
                                if "FALCON" not in text:
                                    text_temp=[int(s) for s in text.split('-' or ' ') if s.isdigit()]
                                    if(len(text_temp) >0):
                                      #  print(text_temp)
                                        text=text_temp[0]

                                        sat_lat.append(np.rad2deg(sat.sublat))
                                        sat_lon.append(np.rad2deg(sat.sublong))
                                        nodes.append([c,sat_lat[c],sat_lon[c],1])
                                        c+=1
                                        sat_name.append(text)

                        except ValueError as e:
                            print(e)

                if(q==0):
                    open_distaces_file(f_name, sats,distance_csv_file)
                write_distences_to_file(f_name,sat_lat,sat_lon,lastHourDateTime,distance_csv_file)

                if (lat_lon__csv_file == True):
                    if (q == 0):
                        open_lat_lon_file(f_lat_lon_name, sat_lat, sat_lon)
                    write_lat_lon_file(f_lat_lon_name, sat_lat, sat_lon, lastHourDateTime)

                t_sat_lat_lon.append([sat_lat, sat_lon])
                t_noeds.append(nodes)
                t_names.append(sat_name)

    dfile = open(f_name, 'ab')
    pickle.dump(distances, dfile)
    dfile.close()
    llfile = open(f_lat_lon_name, 'ab')
    pickle.dump(t_sat_lat_lon, llfile)
    llfile.close()
# if you want files for the VG object
    if (VG_files==True):
        vgFilenodes = open(f_name+"_nodes", 'ab')
        pickle.dump(t_noeds, vgFilenodes)
        vgFilenodes.close()

        vgFilenames = open(f_name + "_names", 'ab')
        pickle.dump(t_names, vgFilenames)
        vgFilenames.close()

        print("files created",f_name+"_nodes",f_name + "_names" )



def open_distaces_file(name, sats,csv=True):
    if csv==True:
        f = open(name+".csv", "w")
        f.write("time")
        for i in range(1,len(sats)-falcons,seperations):
            for j in range(i+1,len(sats)-falcons,seperations):
                f.write(","+str(i)+"_"+str(j))
                distances[str(i)+"_"+str(j)]=[]
        f.write("\n")
        f.close()
    else: #add only distances
        for i in range(1,len(sats)-falcons,seperations):
            for j in range(i+1,len(sats)-falcons,seperations):
                distances[str(i)+"_"+str(j)]=[]


#stas is a list of sats
def write_distences_to_file(name, sat_lat,sat_lon, time, csv=True):
    if csv==True:
        f = open(name+".csv", "a")
        f.write(str(time))

        for i in range(1,len(sat_lat)-falcons,seperations):
            for j in range(i+1,len(sat_lat)-falcons,seperations):
                #d = ephem.separation(sats[i], sats[j])
                #d=math.degrees(float(d))
                d=distance_functions.distance4(sat_lat[i],sat_lat[j], sat_lon[i],sat_lon[j])
                distances[str(i)+"_"+str(j)].append(d)
                f.write(","+str(d))

        f.write("\n")
        f.close()
        print("wrote line to file")
    else:
        for i in range(1,len(sat_lat)-falcons,seperations):
            for j in range(i+1,len(sat_lat)-falcons,seperations):
                d=distance_functions.distance4(sat_lat[i],sat_lat[j], sat_lon[i],sat_lon[j])
                distances[str(i)+"_"+str(j)].append(d)



def write_lat_lon_file(name, lat,lon,time):
    f = open(name+".csv", "a")
    f.write(str(time))
    for i in range(1,len(lat)-falcons,seperations):
        f.write("," + str(lat[i]) + "_" + str(lon[i]))
    f.write("\n")
    f.close()

def open_lat_lon_file(name,  lat,lon):
    f = open(name+".csv", "w")
    f.write("time")
    for i in range(1,len(lat)-falcons,seperations):
        f.write("," + str(i))

    f.write("\n")
    f.close()

#print the list of items in the file
def TestFileCreate(node_f_name, names_f_node, disances_f_name, numOfSat):
    visibilityGraph_for_AC.CreateVGFromFile(node_f_name,names_f_node,disances_f_name, num_of_sats=numOfSat)

def testGraphIst(filename):
    f=open(filename,'rb')
    gl = pickle.load(f)
    f.close()
    costs=[]

    for g in gl:
        sum = 0
        t=dict(nx.all_pairs_dijkstra_path_length(g.vg))
        for key in t:
            for k in t[key]:
                sum+=t[key][k]
        costs.append(sum/(g.vg.number_of_edges()-g.vg.number_of_nodes()))
        print(g.vg.size(weight="weight"))
    print (costs)
    plt.plot(costs)
    plt.show()


    gl[0].draw_graph(draw=True)
    gl[5].draw_graph(draw=True)
    gl[10].draw_graph(draw=True)
    print(len(gl[0].nodes))



def main():
    TestFileCreate("distances_5_10_names" ,"distances_5_10_nodes","distances_5_10",numOfSat=300 )
    testGraphIst("distances_5_10_names_graphs")
    #createFilesFromTLe(5,10,False,False,VG_files=True)

if __name__ == "__main__":
    main()