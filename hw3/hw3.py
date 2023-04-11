"""
第3次作业, 请不要修改输入输出的数据类型和格式
"""
import functools as fs




def choose_best(init_num, count, ans):
    # init_num 为最开始的ans里面个数
    actual_ans = ans[0:init_num]
    # 从最后一个开始，层次遍历，直到回到最初
    st = []  # 实现一个栈
    Q = [ans[-1]]  # 队列
    while len(Q):
        siz = len(Q)
        while siz:
            siz -= 1
            p = Q[0]
            del Q[0]
            st.append(p)

            # 取出里面的数字
            index_1 = p.find('[')
            index_2 = p.find(']')
            lt_p = p[index_1+1:index_2].split(',')
            p1 = int(lt_p[0])
            p2 = int(lt_p[1])

            if p2 > init_num:
                Q.append(ans[p2 - 1])
            if p1 > init_num:
                Q.append(ans[p1 - 1])

    mp = {}  # 用于修改序号数
    for i in range(len(st)-1, -1, -1):
        init_num += 1
        space = st[i].find(' ')
        first_num = int(st[i][0:space])
        mp[first_num] = str(init_num)

        # 分离数字
        index_1 = st[i].find('[')
        index_2 = st[i].find(']')
        lt_p = st[i][index_1 + 1:index_2].split(',')
        p1 = int(lt_p[0])
        p2 = int(lt_p[1])

        cxk1 = lt_p[0]
        cxk2 = lt_p[1]
        for keys, values in mp.items():
            if p1 == keys:
                cxk1 = values
            if p2 == keys:
                cxk2 = values
        if int(cxk1) > int(cxk2):
            t = cxk1
            cxk1 = cxk2
            cxk2 = t
        actual_ans.append(str(init_num) + st[i][space:index_1+1] + cxk1 + ',' + cxk2 + st[i][index_2:])
    return actual_ans


def ResolutionProp(KB):
    """
    :param KB: set(tuple(str))
    :return: list[str]
    """
    ans = []
    KB_cp = [[]]
    # 先全部转换为列表
    count = 1
    # 初始化
    for ele in KB:
        KB_cp.append(list(ele))
        ans.append(str(str(count) + ' ('))
        thislen = len(KB_cp[count])
        for i in range(thislen):
            ans[count - 1] += "'" + str(KB_cp[count][i]) + "'"
            if i != thislen - 1 or i == 0:
                ans[count - 1] += ','
        ans[count - 1] += ')'
        count += 1
    init = count - 1
    i = 1
    while i != count:
        # 枚举用来匹配的字符串
        # newele 用于生成新一组
        newele = []
        # flag用于判断KB_cp[i]与KB_cp[j]是否匹配成功
        flag = 0
        j = 0
        killme = ''
        for str1 in KB_cp[i]:
            tar = ''
            if str1[0] == '~':
                tar = str1[1:]
            else:
                tar = '~' + str1
            # tar为目标查找字符串
            if flag == 0:
                newele.append(str1)
                for j in range(1, count):
                    if tar in KB_cp[j]:
                        # 配对成功，进行合并，也就是KB_cp[i]与KB_cp[j]进行合并
                        # 删去已经匹配的字符串
                        killme = tar
                        del newele[-1]
                        flag = 1
                        # 合并KB_cp[j]
                        break
            else:
                newele.append(str1)

        if flag == 1:
            # 已经推出0了
            if len(newele) == 0 and len(KB_cp[j]) == 1:
                ans.append(str(count) + ' R[' + str(i) + ',' + str(j) + "]: ()")
                actual_ans = choose_best(init, count, ans)
                return actual_ans
            # 把KB_cp[j]剩余的字符串放进去
            for last in KB_cp[j]:
                if last == killme:
                    continue
                newele.append(last)
            ans.append(str(str(count) + ' R[' + str(i) + ',' + str(j) + "]: ("))
            for ele in range(len(newele)):
                ans[count - 1] += "'" + newele[ele] + "'"
                if len(newele) == 1 or ele != len(newele) - 1:
                    ans[count - 1] += ','
            ans[count - 1] += ')'
            count += 1
            KB_cp.append(newele)
        i += 1
    return ans


def is_has_s(s, isvar):
    # s是目标检测字符串，dic是目前字典
    # 用于检测s中是否含有isvar
    if s.find(isvar) == -1:
        # 没有
        return 1
    else:
        return 0


def MGU(f1, f2):
    """
    :param f1: str
    :param f2: str
    :return: dict
    """
    variable = {'xx', 'yy', 'zz', 'uu', 'vv', 'ww'}
    ans = {}
    f1_ele = []
    f2_ele = []
    for i in range(len(f1)):
        if f1[i] == '(':
            f1_ele = f1[i+1:-1].split(',')
            break
    for i in range(len(f2)):
        if f2[i] == '(':
            f2_ele = f2[i+1:-1].split(',')
            break
    # 判断是否常量等
    # 思路，遍历找出常量，消除，再进行循环
    flag = 1

    for i in range(len(f1_ele)):
        isvar = ''
        another_var = ''
        # 多种情况，一是两者之间其中一个是变量，二是他们套在函数里面
        if f1_ele[i] in variable:
            isvar = f1_ele[i]
            another_var = f2_ele[i]
        elif f2_ele[i] in variable:
            isvar = f2_ele[i]
            another_var = f1_ele[i]
        else:
            min_left = [0 for i in range(2)]
            # 找最小括号数，消去最少的部分
            left_start = [0 for i in range(2)]
            for j in range(len(f1_ele[i])):
                if f1_ele[i][j] == '(':
                    min_left[0] += 1
                    left_start[0] = j + 1

            for j in range(len(f2_ele[i])):
                if f2_ele[i][j] == '(':
                    min_left[1] += 1
                    left_start[1] = j + 1
            min_kuo = min(min_left)
            if min_left[0] <= min_left[1]:
                isvar = f1_ele[i][left_start[0]:-min_left[0]]
                another_var = f2_ele[i][left_start[0]:-min_left[0]]
            else:
                isvar = f2_ele[i][left_start[1]:-min_left[1]]
                another_var = f1_ele[i][left_start[1]:-min_left[1]]
        # 出现了变量重复的情况
        if isvar == '':
            return {}
        if is_has_s(another_var, isvar):
            # 变量有多个取值，不能合一
            if ans.get(isvar, 1) == 1:
                ans[isvar] = another_var
                # 接下来更新Sk里的值
                for j in range(i+1, len(f1_ele)):
                    f1_ele[j] = f1_ele[j].replace(isvar, another_var)
                    f2_ele[j] = f2_ele[j].replace(isvar, another_var)
                # 还要更新字典里的
                for key in ans.keys():
                    if ans[key].find(isvar) != -1:
                        # 更新
                        ans[key] = ans[key].replace(isvar, another_var)
            else:
                return {}
        else:
            return {}

    return ans


def ResolutionFOL(KB):
    """
    :param KB: set(tuple(str))
    :return: list[str]
    """
    ans = []
    KB_cp = [[]]
    # 先全部转换为列表
    count = 1
    variable = ['xx', 'yy', 'zz', 'uu', 'vv', 'ww']
    # 用于存放常数

    for ele in KB:
        KB_cp.append(list(ele))
    KB_cp.sort()
    # 初始化
    for i in range(len(KB_cp)-1):
        ans.append(str(str(count) + ' ('))
        thislen = len(KB_cp[count])
        for i in range(thislen):
            ans[count - 1] += "'" + str(KB_cp[count][i]) + "'"
            if i != thislen - 1 or i == 0:
                ans[count - 1] += ','
        ans[count - 1] += ')'
        count += 1

    init = count -1
    i = 1
    while i != count:
        for str1 in KB_cp[i]:
            # 获取谓词
            p1 = 0
            for p1 in range(len(str1)):
                if str1[p1] == '(':
                    break
            predicate1 = str1[0:p1]
            # 取反
            f = 0  # 哨兵，用于判断str1是n还是y
            if predicate1[0] == '~':
                predicate1 = predicate1[1:]
            else:
                f = 1
                predicate1 = '~' + predicate1

            for j in range(0, count):
                for str2 in KB_cp[j]:
                    p2 = 0
                    for p2 in range(len(str2)):
                        if str2[p2] == '(':
                            break
                    predicate2 = str2[0:p2]
                    if predicate1 == predicate2:
                        # 匹配成功 合并两谓词 返回字典，然后开始替换
                        book = {}
                        if f == 0:
                            book = MGU(str1[1:], str2)
                        else:
                            book = MGU(str1, str2[1:])
                        # book为空说明两个都是常量，相等才能消除，否则不行
                        if book == {} and str1[p1+1:] != str2[p2+1:]:
                            continue
                        # 不能用变量替换变量啊啊啊
                        is_vae = 0
                        for values in book.values():
                            if values in variable:
                                is_vae = 1
                                break
                        if is_vae == 1:
                            break
                        # KB_cp[i] 与 KB_cp[j] 可以产生一个孩子，开始替换
                        newele = []

                        for rep_i in KB_cp[i]:
                            if rep_i == str1:
                                continue
                            cp_rep_i = rep_i
                            for key in book.keys():
                                # 将KB_cp[i]中的所有在book中的变量全部替换并且加入newele
                                index = rep_i.find(key)
                                if index != -1:
                                    cp_rep_i = cp_rep_i[0:index] + str(book[key]) + cp_rep_i[index+len(key):]

                            if cp_rep_i not in newele:
                                newele.append(cp_rep_i)

                        for rep_j in KB_cp[j]:
                            if rep_j == str2:
                                continue
                            cp_rep_j = rep_j
                            for key in book.keys():
                                # 将KB_cp[j]中的所有在book中的变量全部替换并且加入newele
                                index = rep_j.find(key)
                                if index != -1:
                                    cp_rep_j = cp_rep_j[0:index] + str(book[key]) + cp_rep_j[index+len(key):]
                            if cp_rep_j not in newele:
                                newele.append(cp_rep_j)

                        # 去重操作，如果子句是永真则去除
                        is_f = 1  # 判断内部是否重复，因为两个常数和两个变量不是永真，一变一常才是永真
                        for d1 in range(len(newele)):
                            fg = 0
                            cp_1 = newele[d1][0:newele[d1].find('(')]
                            in_1 = newele[d1][newele[d1].find('('):]
                            if cp_1[0] == '~':
                                cp_1 = cp_1[1:]
                            else:
                                fg = 1
                                cp_1 = '~' + cp_1
                            for d2 in range(d1+1, len(newele)):
                                cp_2 = newele[d2][0:newele[d2].find('(')]
                                in_2 = newele[d2][newele[d2].find('('):]
                                if cp_1 == cp_2:  # 相反谓词，开始判定
                                    if fg == 0:
                                        lucky = MGU(newele[d1][1:], newele[d2])
                                        if len(lucky) != 0 or in_2 == in_1:
                                            is_f = 0  # 永真
                                            break
                                    else:
                                        lucky = MGU(newele[d1], newele[d2][1:])
                                        if len(lucky) != 0 or in_2 == in_1:
                                            is_f = 0  # 永真
                                            break
                                if is_f == 0:
                                    break
                            if is_f == 0:
                                break
                        if is_f == 0:
                            break

                        # 然后将newele里面的元素直接加入ans中
                        if not newele:
                            ans.append(str(count) + ' R[' + str(i) + ',' + str(j) + ']{}: ()')
                            # 待改
                            return choose_best(init, count, ans)
                        else:
                            # 去重
                            if newele in KB_cp:
                                break
                            KB_cp.append(newele)
                            ans.append(str(count) + ' R[' + str(i) + ',' + str(j) + ']{')
                            for key, values in book.items():
                                ans[count-1] += str(key) + '=' + str(values) + ','
                            if len(book) != 0:
                                ans[-1] = ans[-1][0:-1]  # 除去最后一个','
                            ans[-1] += '}: ('
                            for pp in range(len(newele)):
                                ans[-1] += "'" + str(newele[pp]) + "'"
                                if pp != len(newele) - 1 or pp == 0:
                                    ans[-1] += ','
                            ans[-1] += ')'
                            count += 1
                        break
        i += 1
    return ans



if __name__ == '__main__':
    # 测试程序
    KB1 = {('FirstGrade',), ('~FirstGrade', 'Child'), ('~Child',)}
    result1 = ResolutionProp(KB1)

    for r in result1:
        print(r)

    print(MGU('P(yy,yy)', 'P(xx,a)'))
    print(MGU('P(xx,a,f(g(yy)))', 'P(f(zz),zz,f(uu))'))
    print(MGU('P(x,x)', 'P(y,f(y))'))
    print(MGU('P(f(a),g(x))', 'P(y,y)'))
    KB3 = {('A(tony)',), ('A(mike)',), ('A(john)',), ('L(tony,rain)',), ('L(tony,snow)',), ('~A(xx)', 'S(xx)', 'C(xx)'),
           ('~C(yy)', '~L(yy,rain)'), ('L(zz,snow)', '~S(zz)'), ('~L(tony,uu)', '~L(mike,uu)'),
           ('L(tony,vv)', 'L(mike,vv)'), ('~A(ww)', '~C(ww)', 'S(ww)')}
    KB2 = {('On(a,b)',), ('On(b,c)',), ('Green(a)',), ('~Green(c)',), ('~On(xx,yy)', '~Green(xx)', 'Green(yy)')}
    # KB2 = {('P(xx)',),('~P(a)','Q(b)'),('~P(c)','R(xx)','S(e)'),('~R(c)','S(e)'),('~S(e)',)}
    result2 = ResolutionFOL(KB3)
    for r in result2:
        print(r)


