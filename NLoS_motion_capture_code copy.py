import os
import time

lt = ["nlos_walking","nlos_standing","nlos_sitting","nlos_running"]

for kk in range(len(lt)):
    for jj in range(0,100):
        print('start' + str(lt[kk]) + str(jj))
        os.system('sudo tcpdump -i wlan0 src 10.10.10.10 -w /home/pi/py_code/'+str(lt[kk])+str(jj)+'.pcap -c 10')
        print('---------------------------------------')
        print('done' + str(lt[kk]) + str(jj))
        print('---------------------------------------')
    print('clear')
    time.sleep(10)

print('end')




