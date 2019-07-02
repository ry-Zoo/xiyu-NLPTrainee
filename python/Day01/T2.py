def GenerateArea(output_str,left_index,right_index,rank):
	if right_index==rank and left_index==rank:
		ls.append(output_str)
	else:
		if left_index < rank :
			GenerateArea(output_str+'(',left_index+1,right_index,rank)
		if right_index < left_index :
			GenerateArea(output_str+')',left_index,right_index+1,rank)


ls=[]
GenerateArea("",0,0,3)
for item in ls:
    print (item+'\n')

		