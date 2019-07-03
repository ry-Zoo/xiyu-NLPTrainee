class Solution(object):
    def longestCommonPrefix(self, strs):
        if not strs:
            return ''
        s1 = min(strs)
        print(s1)
        s2 = max(strs)
        print(s2)
        for i, c in enumerate(s1):

            if c != s2[i]:
                return s1[:i]
        return s1


if __name__ == '__main__':
    s = Solution()
    #print(s.longestCommonPrefix(["flower", "flow", "flight"]))
    print(s.longestCommonPrefix(["far", "racecar", "car"]))