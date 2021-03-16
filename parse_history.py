#preprocessing of voter regristration and voter history files
#enable batch processing by only keeping relevant records
import csv

data = []

f = open("data/ncvhis_Statewide.txt")
g = open('data/ncvhis.csv', 'a', newline='')
write = csv.writer(g)

#first line is headers, clear that
f.readline()
i = 0
a=0
while f:
    temp = []
    #get first row of data
    s = f.readline()
    
    #break at end of file
    if s == "":
        break
    
    #replace all quotation marks
    s = s.replace('\"','')
    #split along tabs
    s = s.split('\t')
    #split the election column to filter against general and primary elections
    filt = s[4].split(' ',1)[1]
    #if it is a general or primary election keep
    if filt == 'GENERAL' or filt == 'PRIMARY':
        #keep columns 0,2,3,4 for later use
        for j in range(len(s)):
            if j == 0 or j == 2 or j == 3 or j == 4:
                #append columns to temp list
                temp.append(s[j])
        #append list to variable for write
        data.append(temp)
    
    if i == 100000:
        #at 100000 iterations save data to file
        write.writerows(data)
        #clear data variable and repeat
        data = []
        i=0
        a+=1
    
    #break at 100,000,000 rows (only ~40,000,000 in file)
    if a > 1000:
        break
    
    i+=1

f.close()
g.close()
