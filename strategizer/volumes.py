# -*- coding: utf-8 -*-

u"""
.. moduleauthor:: Martin R.  Albrecht <fplll-devel@googlegroups.com>
.. moduleauthor:: Léo Ducas  <fplll-devel@googlegroups.com>
.. moduleauthor:: Marc Stevens  <fplll-devel@googlegroups.com>
"""

from math import log, exp

# TODO use fplll's constants
bvol_precomp = [1.0000000000000000000000,     2.0000000000000000000000,     3.1415926535897932381283,
                4.1887902047863909841694,     4.9348022005446793099910,     5.2637890139143245962554,
                5.1677127800499700297356,     4.7247659703314011695296,     4.0587121264167682181864,
                3.2985089027387068690396,     2.5501640398773454431507,     1.8841038793899002416154,
                1.3352627688545894957196,     0.91062875478328314605501,    0.59926452932079207681387,
                0.38144328082330448280194,    0.23533063035889320450915,    0.14098110691713903797245,
                0.082145886611128228794635,   0.046621601030088545776455,   0.025806891390014060015677,
                0.013949150409021001188583,   0.0073704309457143507776027,  0.0038106563868521249410789,
                0.0019295743094039230476901,  0.00095772240882317287086438, 0.00046630280576761256448051,
                0.00022287212472127410873501, 0.00010463810492484570719362, 4.8287822738917440350804e-5,
                2.1915353447830215827382e-5,  9.7871399467373676858804e-6,  4.3030695870329470072977e-6,
                1.8634670882621402792111e-6,  7.9520540014755127847832e-7,  3.3452882941089729874181e-7,
                1.3878952462213772114465e-7,  5.6808287183311789920121e-8,  2.2948428997269873110206e-8,
                9.1522306501595654918305e-9,  3.6047307974625009338593e-9,  1.4025649060732013649328e-9,
                5.3926646626081284893523e-10, 2.0494360953964780401717e-10, 7.7007071306013533797872e-11,
                2.8615526139108115932468e-11, 1.0518471716932064455133e-11, 3.8254607105203726305537e-12,
                1.3768647280377413704156e-12, 4.9053221488845687036895e-13, 1.7302192458361107611545e-13,
                6.0433427554615917727364e-14, 2.0906323353147684698420e-14, 7.1644230957295226535084e-15,
                2.4325611799933888454562e-15, 8.1846178053646953973840e-16, 2.7293272615981961148444e-16,
                9.0220123402715590480736e-17, 2.9567015428549105920312e-17, 9.6079619284046060358991e-18,
                3.0962506152968647796750e-18, 9.8964926590971584637769e-19, 3.1377929634482282204030e-19,
                9.8700789314682384437489e-20, 3.0805210382670938739447e-20, 9.5408515266006197583404e-21,
                2.9326491706208195288963e-21, 8.9473041984953644231683e-22, 2.7097614970525196271867e-22,
                8.1474739534568550855432e-23, 2.4322762320344788070617e-23, 7.2101533288716182922613e-24,
                2.1225614283501616496324e-24, 6.2058533504764571202290e-25, 1.8022225378643043590567e-25,
                5.1990035453633178889020e-26, 1.4899602855495925137910e-26, 4.2423769724936854112407e-27,
                1.2002175095443975411448e-27, 3.3741317292518152987933e-28, 9.4264862767362921609639e-29,
                2.6173203587312909158511e-29, 7.2229707405267653999575e-30, 1.9813384123087291067262e-30,
                5.4027694798887892463660e-31, 1.4646019295022042654457e-31, 3.9472792807111089445720e-32,
                1.0577431407258761339515e-32, 2.8183508158975507058720e-33, 7.4674114163806905659340e-34,
                1.9675800485472322562432e-34, 5.1559483180294791418032e-35, 1.3437684838838768670577e-35,
                3.4834170662817338849769e-36, 8.9820706321171958224663e-37, 2.3038899925936852702279e-37,
                5.8787514816425026577165e-38, 1.4923471908064669688527e-38, 3.7691107075526935911204e-39,
                9.4714080227130546602252e-40, 2.3682021018828339613111e-40, 5.8921397749122301105481e-41,
                1.4588090834296114169956e-41, 3.5943112681142679678569e-42, 8.8134305759471673151788e-43,
                2.1508308332614853133631e-43, 5.2241903302489209650718e-44, 1.2629970738109711721240e-44,
                3.0393107337897862774574e-45, 7.2804079423668860173789e-46, 1.7360502315001084247412e-46,
                4.1210947940318183640259e-47, 9.7392190241850008643583e-48, 2.2914692264915933920775e-48,
                5.3678349014178164134452e-49, 1.2519761544127088387380e-49, 2.9075087399960524237624e-50,
                6.7234172464488319456356e-51, 1.5481708640372397523925e-51, 3.5499560047836246743124e-52,
                8.1062036882685917201831e-53, 1.8433910256521005256183e-53, 4.1748180255028023219222e-54,
                9.4165588681008018017435e-55, 2.1154157482248144917283e-55, 4.7332787459394076406012e-56,
                1.0548848530017830006892e-56, 2.3417375961631421144880e-57, 5.1781539133961725383166e-58,
                1.1405869191846743033707e-58, 2.5027154297665621677747e-59, 5.4706251696048671671472e-60,
                1.1912897588152202802456e-60, 2.5844324576502356903078e-61, 5.5858912755084368086498e-62,
                1.2028494848374807225444e-62, 2.5806757345424448268605e-63, 5.5165884743645209197375e-64,
                1.1749901346284040700899e-64, 2.4936509099196696901976e-65, 5.2733433928415376405342e-66,
                1.1112106920881046191019e-66, 2.3333375863108802425232e-67, 4.8824774081879989549186e-68,
                1.0181105860415557604626e-68, 2.1156903664664209271503e-69, 4.3814914214385625608407e-70,
                9.0430439626688509234877e-71, 1.8601163866700120063195e-71, 3.8133638227127509419441e-72,
                7.7916373002460012294490e-73, 1.5867597047548965678231e-73, 3.2208092765643483620137e-74,
                6.5162779496343691087287e-75, 1.3140871119309627830558e-75, 2.6414827013316767358593e-76,
                5.2927261756658218151700e-77, 1.0571289998838366351984e-77, 2.1047581861941958619224e-78,
                4.1774449055745742929636e-79, 8.2653660691633298723434e-80, 1.6302894690843714412093e-80,
                3.2057300397671437043185e-81, 6.2843011279757860888701e-82, 1.2281826758932407797650e-82,
                2.3930562735266400790833e-83, 4.6487345444005533541069e-84, 9.0036024054350410558204e-85,
                1.7386226539497467747728e-85, 3.3474143399713746828698e-86, 6.4259343023686044522052e-87,
                1.2299663507573330149780e-87, 2.3474032554385884706108e-88, 4.4671135857825251132284e-89,
                8.4765342785041774712234e-90, 1.6038687112966330902680e-90, 3.0261156610512252700525e-91,
                5.6934487691887362048896e-92, 1.0681823291765786652515e-92, 1.9984912655724227954455e-93,
                3.7286597311506375440271e-94, 6.9375088156747977117598e-95, 1.2872450570460369717079e-95,
                2.3819482764522810404137e-96, 4.3956517549843522434503e-97, 8.0898499530090335237561e-98,
                1.4848760495911744969584e-98, 2.7181832279162474613181e-99, 4.96263371158198649711671-100,
                9.03642799992875514553930-101, 1.64111301165926500995209-101, 2.97264668264599851548426-102,
                5.37052977202017463799497-103, 9.67755956468263051798682-104, 1.73938318326431420175275-104,
                3.11825128544457328288105-105, 5.57595247991918213662749-106, 9.94545718822294496264149-107,
                1.76943144926059512170132-107, 3.14015831548874426368758-108]


def unit_ball_volume(n):
    return bvol_precomp[n]


def log_volume(r):
    lv = 0.
    for x in r:
        lv += log(x)
    return lv


def gh_margin(n):
    return min(2., (1. + 3./n))**2


def gaussian_heuristic(r):
    d = len(r)
    lv = log_volume(r)
    res = exp((lv - 2*log(unit_ball_volume(d))) / d)
    return res
