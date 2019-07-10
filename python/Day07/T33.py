'''
import numpy as np
def minSubstr(raw_str,enum_str):
	
	index_dict={}
	for j in range(0,len(raw_str)):
		for i in range(0,len(enum_str)):
			if enum_str[i]==raw_str[j]:
				index_dict[enum_str[i]].append(j)
	u_list=[]
	for item in index_dict.values:
		u_list.append(item)
	for 

def variance(ls):
	variance=np.std(ls,axis=0)
	return variance
'''
#尝试使用决策表+方差方法，发现方差计算时候计算量根据给定字母表的长度程幂次增长

'''
from collections  import Counter

def minSubstr(raw_str,enum_str):
	all_dic=Counter(raw_str)
	enum_dic={}
	for e in all_dic.keys():
		if e in enum_str:
			enum_dic[e]=all_dic[e]
	left_curse=0
	right_curse=len(raw_str)
	slide=[1 for i in range(0,len(enum_dic))]	
	#print(list(enum_dic.values()))
	#print(slide)
	while slide!=list(enum_dic.values()):
		print(list(enum_dic.values()))
		if raw_str[left_curse] in enum_dic.keys():
			if 0 not in  list(enum_dic.values()):
				enum_dic[raw_str[left_curse]]-=1
				if enum_dic[raw_str[left_curse]]==0:
					enum_dic[raw_str[left_curse]]+=1
				else:
					left_curse+=1
			else:
				break
		if raw_str[right_curse-1] in enum_dic.keys():
			if 0 not in  list(enum_dic.values()):
				enum_dic[raw_str[right_curse-1]]-=1
				if enum_dic[raw_str[right_curse-1]]==0:
					enum_dic[raw_str[right_curse-1]]+=1
				else:
					right_curse-=1
			else:
				break
		else:
			left_curse+=1
			right_curse-=1
	print(raw_str[left_curse:right_curse])


minSubstr('ADOBECODEBANC','ABC')

'''
#尝试从两侧接近，截取最小长度的子串，失败
#
#
def minWindow(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: str
    """
    if not t or not s:
        return ""

    dict_t = Counter(t)

    required = len(dict_t)

    # Filter all the characters from s into a new list along with their index.
    # The filtering criteria is that the character should be present in t.
    filtered_s = []
    for i, char in enumerate(s):
        if char in dict_t:
            filtered_s.append((i, char))

    l, r = 0, 0
    formed = 0
    window_counts = {}

    ans = float("inf"), None, None

    # Look for the characters only in the filtered list instead of entire s. This helps to reduce our search.
    # Hence, we follow the sliding window approach on as small list.
    while r < len(filtered_s):
        character = filtered_s[r][1]
        window_counts[character] = window_counts.get(character, 0) + 1

        if window_counts[character] == dict_t[character]:
            formed += 1

        # If the current window has all the characters in desired frequencies i.e. t is present in the window
        while l <= r and formed == required:
            character = filtered_s[l][1]

            # Save the smallest window until now.
            end = filtered_s[r][0]
            start = filtered_s[l][0]
            if end - start + 1 < ans[0]:
                ans = (end - start + 1, start, end)

            window_counts[character] -= 1
            if window_counts[character] < dict_t[character]:
                formed -= 1
            l += 1    

        r += 1    
    return "" if ans[0] == float("inf") else s[ans[1] : ans[2] + 1]

