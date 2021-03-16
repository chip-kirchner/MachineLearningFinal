#preprocessing of voter regristration files
#move line by line to not exhaust RAM
import csv

#placeholder for write data
data = []

f = open("data/ncvoter_Statewide.txt")
g = open('data/ncvreg.csv', 'a', newline='')
write = csv.writer(g)

#first line is headers skip these
f.readline()
i = 0
a=0
while f:
    temp = []
    #get row of data
    s = f.readline()
    
    #break at end of file
    if s == "":
        break
    
    #replace quotation marks and split along tabs
    s = s.replace('\"','')
    s = s.split('\t')
    
    #only keep active voters (help reduce file size for model use)
    if s[3] == 'A':
        for j in range(len(s)):
            #keep columns specified in paper
            if j == 0 or j == 2 or (j > 12 and j<17) or (j>24 and j<33):
                #append to temp holder for line
                temp.append(s[j])
        #append to hopper for write
        data.append(temp)
    
    if i == 100000:
        #at 100,000 iteration write data to file
        write.writerows(data)
        #clear and repeat
        data = []
        i=0
        a+=1
    
    #at 100,000,000 rows break, (less than 10,000,000 in file)
    if a > 1000:
        break
    
    i+=1

f.close()
g.close()
