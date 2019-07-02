def isValid (str):
	list =  []
	for c in str :
		if (c=="("):
			list.append(")")
		elif (c=="["):
			list.append("]")
		elif (c =="{"):
			list.append("}")
		elif (not list or c!=list.pop()):
			print ("False")
	print ("True")
	return True
	


isValid("(([()])){}")