import argparse
            
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size','-batch_size',required=True,type=int,help='batch_size')
args = parser.parse_args()
batchsize=args.batch_size


filepath ="/home/kugua/dongd/sequence/all_Up_seq"

dicc = {}
dicc['A'] = '1\t0\t0\t0'
dicc['C'] = '0\t1\t0\t0'
dicc['G'] = '0\t0\t1\t0'
dicc['T'] = '0\t0\t0\t1'
dicc['N'] = '0\t0\t0\t0'
fout1 = open("data/trainsequence0", 'w')
fout2 = open("data/trainlabels0", 'w')
fout12 = open("data/trainy0", 'w')
fout5 = open("data/trainlength0", 'w')
fout3 = open("data/testsequence0", 'w')
fout4 = open("data/testlabels0", 'w')
fout14 = open("data/testy0", 'w')
fout6 = open("data/testlength0", 'w')
fout7 = open("data/validationsequence0", 'w')
fout8 = open("data/validationlabels0", 'w')
fout18 = open("data/validationy0", 'w')
fout9 = open("data/validationlength0", 'w')

for i in range(3999):
    fout3.write(str(i)+"\t")
fout3.write("3999\n")
fout4.write("1\t2\n")
fout14.write("1\t2\n")
fout6.write("1\n")
for i in range(3999):
    fout7.write(str(i)+"\t")
fout7.write("3999\n")
fout8.write("1\t2\n")
fout18.write("1\t2\n")
fout9.write("1\n")

number = 0
fin = open(filepath, 'r')
line = fin.readline()
index=0
while line and line!="\n":
    seq = fin.readline()
    seq = seq.strip('\n') 
    number += 1
    if number>624481 and number<=646481:
        for i in seq[:-1]:
            fout3.write(dicc[i]+"\t")
        fout3.write(dicc[seq[-1]]+"\t")
        for i in range(1,1000-len(seq)):
            fout3.write(dicc['N']+"\t")
        fout3.write(dicc['N']+"\n")
        if line[0]=='p':
            fout4.write("0\t1\n")
            fout14.write("0\t1\n")
        else:
            fout4.write("1\t0\n")
            fout14.write("1\t0\n")
        fout6.write(str(len(seq))+"\n")
    if number<=613481:
        if number%batchsize==1:
            index=number//batchsize
            fout1.close()
            fout2.close()
            fout5.close()
            fout1 = open("data/trainsequence"+str(index), 'w')
            fout2 = open("data/trainlabels"+str(index), 'w')
            fout12 = open("data/trainy"+str(index), 'w')
            fout5 = open("data/trainlength"+str(index), 'w')
            for i in range(3999):
               fout1.write(str(i)+"\t")
            fout1.write("3999\n")
            fout2.write("1\t2\n")
            fout12.write("1\t2\n")
            fout5.write("1\n")
        for i in seq[:-1]:
            fout1.write(dicc[i]+"\t")
        fout1.write(dicc[seq[-1]]+"\t")
        for i in range(1,1000-len(seq)):
            fout1.write(dicc['N']+"\t")
        fout1.write(dicc['N']+"\n")
        if line[0]=='p':
            fout2.write("0\t1\n")
            fout12.write("0\t1\n")
        else:
            fout2.write("1\t0\n")
            fout12.write("1\t0\n")
        fout5.write(str(len(seq))+"\n")
    if number>613481 and number<=624481:
        for i in seq[:-1]:
            fout7.write(dicc[i]+"\t")
        fout7.write(dicc[seq[-1]]+"\t")
        for i in range(1,1000-len(seq)):
            fout7.write(dicc['N']+"\t")
        fout7.write(dicc['N']+"\n")
        if line[0]=='p':
            fout8.write("0\t1\n")
            fout18.write("0\t1\n")
        else:
            fout8.write("1\t0\n")
            fout18.write("1\t0\n")
        fout9.write(str(len(seq))+"\n")
    line = fin.readline()
fout1.close()
fout2.close()
fout12.close()
fout3.close()
fout4.close()
fout14.close()
fout5.close()
fout6.close()
fout7.close()
fout8.close()
fout18.close()
fout9.close()
print("OK")
