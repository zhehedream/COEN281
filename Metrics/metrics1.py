file1 = open('../Mondrian/data/anonymized.data', 'r') 
Lines = file1.readlines()

for line in Lines: 
    print(line.strip()) 
