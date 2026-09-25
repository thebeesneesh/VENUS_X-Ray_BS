import tkinter as tk
from tkinter import messagebox, ttk, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D


# Embedded axial data:
# z_cm, injection profile at 185 A, center profile at -146 A,
# extraction profile at 152 A, sextupole profile at 460 A
AXIAL_DATA = """
-100 28.65340139 -1.730542893 1.209372758 0.000565649
-99 31.46335425 -1.89525235 1.313677926 0.000605266
-98 34.27330712 -2.059961808 1.417983094 0.000644883
-97 37.08325998 -2.224671265 1.522288263 0.000684499
-96 39.89321285 -2.389380723 1.626593431 0.000724116
-95 42.70316572 -2.554090181 1.730898599 0.000763732
-94 45.51311858 -2.718799638 1.835203768 0.000803349
-93 48.32307145 -2.883509096 1.939508936 0.000842965
-92 51.13302432 -3.048218553 2.043814105 0.000882582
-91 54.98713772 -3.275478068 2.186613543 0.000903663
-90 62.60730695 -3.728341789 2.468253287 0.000857892
-89 70.22747619 -4.18120551 2.749893031 0.000812121
-88 77.84764542 -4.634069231 3.031532775 0.00076635
-87 85.46781466 -5.086932951 3.313172519 0.00072058
-86 93.08798389 -5.539796672 3.594812262 0.000674809
-85 100.7081531 -5.992660393 3.876452006 0.000629038
-84 108.3283224 -6.445524114 4.15809175 0.000583267
-83 115.9484916 -6.898387834 4.439731494 0.000537496
-82 123.5686608 -7.351251555 4.721371238 0.000491725
-81 134.1504769 -7.984216939 5.114067714 0.000462274
-80 150.8079292 -8.986649813 5.734590248 0.000466303
-79 167.4653816 -9.989082688 6.355112782 0.000470331
-78 184.122834 -10.99151556 6.975635317 0.00047436
-77 200.7802864 -11.99394844 7.596157851 0.000478389
-76 224.5005884 -13.43119071 8.484385314 0.000241606
-75 259.5971819 -15.5687903 9.803811159 -0.000383058
-74 294.6937755 -17.7063899 11.123237 -0.001007722
-73 329.790369 -19.84398949 12.44266285 -0.001632386
-72 364.8869626 -21.98158909 13.76208869 -0.00225705
-71 423.8377674 -25.58297356 15.98416248 -0.003392988
-70 513.3516589 -31.05982497 19.36274934 -0.005183994
-69 602.8655504 -36.53667637 22.7413362 -0.006975
-68 722.4712732 -43.88944944 27.27467259 -0.009270431
-67 843.1355528 -51.30821285 31.84863024 -0.011583606
-66 987.17702 -60.17765474 37.31511199 -0.014606327
-65 1155.194352 -70.53492625 43.69697494 -0.018356764
-64 1325.541471 -81.03679906 50.1675337 -0.022325682
-63 1602.737061 -98.17036571 60.70585291 -0.036314542
-62 1842.698243 -113.439317 70.06148385 -0.049501224
-61 2192.277416 -134.8885386 83.27295254 -0.054748809
-60 2620.007965 -160.9968413 99.45599281 0.003381566
-59 3068.963839 -188.7151308 116.5177694 0.014623464
-58 3584.758123 -220.7583866 136.1220481 -0.033823822
-57 4162.240906 -256.3619729 158.0164416 0.000586087
-56 4809.016251 -296.2107023 182.494679 0.050855511
-55 5522.941794 -340.1654879 209.1636035 -0.055692584
-54 6304.190732 -388.0396795 238.3549521 -0.034981387
-53 7123.557862 -437.3798282 268.4132049 0.064533915
-52 8000.318387 -490.1444284 300.0879236 -0.040764925
-51 8913.600358 -544.1303887 332.4780145 -0.025920837
-50 9858.905655 -599.0329065 365.2330388 0.029776775
-49 10827.29049 -654.10173 397.6779078 0.030694865
-48 11812.05694 -708.5941958 429.3557241 0.015446099
-47 12810.53504 -762.1111334 460.0563218 0.042579962
-46 13831.34802 -815.4547517 490.1394435 0.05197798
-45 14871.57262 -868.3216981 519.3221972 0.029242428
-44 15936.08585 -921.1400951 547.9247783 0.04452007
-43 17027.25239 -974.1843675 575.9697537 0.057995011
-42 18148.87238 -1028.176508 603.8063634 0.048973603
-41 19303.97972 -1083.889544 631.8801493 0.040386655
-40 20490.67663 -1141.557804 660.2269825 0.036057559
-39 21707.61101 -1201.889499 689.2090071 0.049813245
-38 22949.82727 -1265.503824 719.0511496 0.054963069
-37 24209.62102 -1333.081153 750.0641623 0.054389412
-36 25476.63309 -1405.216461 782.4957322 0.057542004
-35 26738.12348 -1482.448829 816.5099869 0.057990336
-34 27978.94192 -1565.414572 852.355222 0.060748041
-33 29182.778 -1654.577232 890.1058651 0.062532832
-32 30331.70176 -1750.511536 929.9141961 0.05985044
-31 31406.74744 -1853.897226 972.0347695 0.062910335
-30 32389.63635 -1965.209233 1016.512101 0.060225105
-29 33261.68448 -2085.093938 1063.596348 0.050546367
-28 34006.88601 -2213.855217 1113.129969 0.033638299
-27 34610.07638 -2351.808573 1165.028186 0.01634685
-26 35055.49271 -2499.484181 1219.569611 -0.0000707
-25 35331.50758 -2657.134446 1276.568774 -0.029892957
-24 35427.5217 -2824.812649 1335.983269 0.013402674
-23 35337.84067 -3003.277046 1397.448811 -0.101457472
-22 35053.95965 -3192.304367 1461.070551 -0.256185857
-21 34574.34893 -3392.392196 1526.856573 -0.484414919
-20 33909.35761 -3604.566711 1595.561217 -0.232862082
-19 33064.01823 -3829.804966 1667.206809 0.123304786
-18 32059.06915 -4071.58455 1741.703098 -0.393651818
-17 30912.06066 -4329.980059 1820.477575 -1.010172529
-16 29657.60906 -4606.094417 1906.818954 0.193634777
-15 28324.37574 -4906.050228 1998.082957 -0.74607829
-14 26941.67406 -5226.982379 2098.954106 -0.233770245
-13 25534.66608 -5571.440254 2208.180155 -0.633184838
-12 24128.47931 -5935.894057 2329.3596 0.687440025
-11 22738.91559 -6321.538411 2460.255878 0.567326837
-10 21377.44655 -6725.131805 2600.514812 -1.990759425
-9 20060.46439 -7136.587501 2756.503149 -1.421840837
-8 18794.29579 -7553.187227 2926.671011 -0.307569779
-7 17583.48825 -7966.385307 3111.089765 0.920132035
-6 16429.41271 -8371.479473 3307.66004 -0.999710017
-5 15339.16955 -8754.83793 3522.76299 -0.964288854
-4 14313.61516 -9106.503578 3757.403054 0.837548925
-3 13348.1301 -9420.945767 4008.770225 -0.09566215
-2 12447.48347 -9683.069483 4284.046775 1.932216272
-1 11602.9363 -9891.438382 4577.650149 -0.592304525
0 10819.15454 -10032.60906 4898.225847 -0.798699829
1 10089.86326 -10104.74251 5245.361069 -1.817293719
2 9414.879746 -10102.67795 5624.455898 -0.775645347
3 8791.530409 -10027.15901 6036.924835 1.700864146
4 8211.82291 -9884.371245 6482.592232 1.561857805
5 7673.513878 -9679.915955 6964.124251 -1.121094075
6 7180.561414 -9414.423815 7491.327484 0.51419807
7 6724.279952 -9101.226889 8062.537614 0.994025024
8 6304.17886 -8750.044484 8681.713463 1.504639769
9 5916.809287 -8371.760036 9350.922251 1.042479865
10 5558.976672 -7977.97935 10071.26921 -1.649876295
11 5234.647665 -7571.958234 10849.32781 -0.505255106
12 4935.543412 -7167.859961 11677.14517 -1.870056516
13 4664.924189 -6769.08904 12557.2301 -0.985365994
14 4418.274163 -6382.839084 13479.22418 -0.160194263
15 4192.703639 -6013.45128 14431.46894 -0.018923534
16 3986.22022 -5664.135631 15401.51102 -0.510985418
17 3796.450431 -5332.776714 16368.08686 -0.566400156
18 3622.261433 -5018.388582 17313.67561 1.112447718
19 3456.531015 -4724.339356 18212.91858 -0.004373016
20 3300.597913 -4444.70793 19047.11952 -0.051030112
21 3151.874058 -4179.709421 19798.19179 -0.065077736
22 3009.08464 -3928.476967 20450.43073 -0.156805026
23 2872.009461 -3689.35187 20989.84428 0.14320968
24 2739.94717 -3463.284892 21408.54946 0.249746232
25 2613.139813 -3250.101423 21699.4602 0.16744822
26 2491.599333 -3049.67015 21860.87894 0.144830098
27 2375.70895 -2861.692784 21890.74208 0.132697811
28 2265.403729 -2686.087682 21792.13673 0.126561222
29 2161.089994 -2522.663435 21569.36754 0.116444266
30 2062.057858 -2370.615851 21231.49314 0.125176457
31 1968.968196 -2229.922963 20787.06767 0.095573856
32 1881.115711 -2099.585359 20249.70151 0.093840794
33 1798.442212 -1979.125999 19632.69839 0.082007076
34 1721.159924 -1868.14782 18949.66604 0.076757618
35 1648.277733 -1765.547463 18217.92615 0.072369418
36 1579.652796 -1670.775657 17451.91075 0.060332047
37 1515.359747 -1583.400816 16664.6376 0.056453928
38 1454.797988 -1502.620877 15869.70903 0.053222541
39 1397.722063 -1427.874248 15077.64336 0.048459906
40 1343.611563 -1358.431461 14298.0074 0.043549669
41 1292.268976 -1293.787782 13536.66891 0.043818471
42 1242.908436 -1233.090031 12799.37207 0.034867501
43 1194.981899 -1175.587994 12088.19824 0.035227627
44 1148.214718 -1120.77426 11402.47541 0.044646317
45 1101.572456 -1067.638454 10739.97128 0.028910407
46 1054.810296 -1015.66068 10098.54329 0.043194992
47 1006.379347 -963.4428122 9470.548602 0.031408009
48 955.2932101 -909.8913021 8848.615991 0.028960799
49 901.9817331 -855.1960089 8231.331188 0.04205696
50 844.6606775 -797.715073 7607.269757 0.048829496
51 783.8116701 -737.7535044 6976.774763 0.050627828
52 718.5395256 -674.4680933 6333.852965 0.034437072
53 650.2242509 -608.968467 5685.460639 0.019925549
54 578.973463 -541.3034803 5031.903581 0.001343704
55 510.3909602 -476.3869249 4410.543788 0.028183462
56 441.8213068 -411.868713 3804.436236 0.059851974
57 380.2262961 -354.1056222 3260.097204 0.028486904
58 319.8584776 -297.6888261 2739.113184 0.040385965
59 267.7344102 -249.033082 2287.695612 0.03086974
60 221.129073 -205.6402702 1887.079791 0.000467898
61 180.420947 -167.7131789 1538.072625 0.008604459
62 145.1621903 -134.9028296 1236.674806 0.008619354
63 115.2121278 -107.0885338 981.4179267 -0.014299928
64 87.95629702 -81.75510749 744.612636 -0.00697282
65 69.7589567 -64.8145234 592.6246192 -0.009564019
66 53.59134871 -49.77024808 455.908938 -0.007428185
67 40.62619396 -37.70792287 345.5694151 -0.004781249
68 30.66839976 -28.44255234 260.8073359 -0.003151475
69 23.06039779 -21.36206509 196.0483305 -0.001991033
70 17.44086757 -16.1311802 148.2153679 -0.00123734
71 13.16268325 -12.14794635 111.7960821 -0.000577139
72 10.04357051 -9.243659129 85.23929117 -0.000144201
73 7.73657748 -7.095466317 65.59029864 0.000218512
74 6.070723457 -5.544484137 51.39499104 0.000459893
75 4.83015922 -4.389726807 40.81577681 0.000624229
76 3.863643259 -3.490319271 32.56443598 0.000739445
77 3.155552491 -2.831651735 26.50966448 0.000775385
78 2.623234765 -2.336582182 21.94714717 0.000782578
79 2.202694054 -1.945442719 18.330752 0.000764149
80 1.878103952 -1.643371374 15.52657699 0.000729114
81 1.619138779 -1.401984874 13.27350464 0.000690652
82 1.415816477 -1.211892641 11.48702604 0.000646595
83 1.2482719 -1.054338825 9.992328582 0.000605551
84 1.111483433 -0.924393995 8.743870176 0.000566662
85 1.000753657 -0.817279756 7.696117225 0.000529892
86 0.906442048 -0.723248052 6.753868786 0.000490598
87 0.833277494 -0.646325241 5.956456538 0.000455816
88 0.772783753 -0.576974041 5.205314363 0.000422106
89 0.729176355 -0.51877407 4.536628162 0.000392327
90 0.702827415 -0.470480637 3.932757797 0.000368616
91 0.693231335 -0.430253805 3.370994239 0.000349343
92 0.703223672 -0.401445018 2.890465679 0.000337734
93 0.731213384 -0.382717878 2.479702191 0.000333639
94 0.771980828 -0.373922288 2.160930593 0.00033674
95 0.822308218 -0.372213674 1.906079877 0.000346247
96 0.875297745 -0.375721297 1.718369714 0.000360421
97 0.925578298 -0.381835777 1.582851087 0.000379341
98 0.968136858 -0.388333211 1.488674052 0.000398305
99 1.001901975 -0.39396593 1.421915965 0.000416374
100 1.026478903 -0.397978904 1.37304188 0.000432567
"""


# The supplied radial data follow B = 0.039603 * r^2 at 430 A.
RADIAL_RADIUS_CM = np.arange(0.0, 7.2001, 0.1)
RADIAL_FIELD_430A_T = 0.039603 * RADIAL_RADIUS_CM**2


def load_axial_data():
    data = np.fromstring(AXIAL_DATA, sep=" ").reshape(-1, 5)

    if len(data) != 201:
        raise RuntimeError(f"Expected 201 axial rows, got {len(data)}.")

    return data.T


class VenusBFieldCalculator:
    def __init__(self):
        (
            self.z_cm,
            self.injection_185a,
            self.center_minus_146a,
            self.extraction_152a,
            self.sextupole_460a,
        ) = load_axial_data()

    def calculate_axial(self, injection_a, center_a, extraction_a):
        injection_g = injection_a / 185.0 * self.injection_185a
        center_g = center_a / -146.0 * self.center_minus_146a
        extraction_g = extraction_a / 152.0 * self.extraction_152a

        # The embedded axial sextupole profile is the 460 A profile.
        sextupole_g = self.sextupole_460a

        total_g = injection_g + center_g + extraction_g + sextupole_g

        return {
            "injection_t": injection_g / 10000.0,
            "center_t": center_g / 10000.0,
            "extraction_t": extraction_g / 10000.0,
            "sextupole_t": sextupole_g / 10000.0,
            "total_t": total_g / 10000.0,
        }

    def calculate_radial(self, sextupole_a):
        return RADIAL_FIELD_430A_T * (sextupole_a / 430.0)


class VenusApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("VENUS B-Field Calculator")
        self.geometry("1100x750")

        self.calculator = VenusBFieldCalculator()
        self._axial_plot_data = None
        self._create_tabs()

    @staticmethod
    def _entry(parent, label, value, row):
        ttk.Label(parent, text=label).grid(
            row=row,
            column=0,
            padx=8,
            pady=5,
            sticky="w",
        )

        entry = ttk.Entry(parent, width=14)
        entry.insert(0, str(value))
        entry.grid(row=row, column=1, padx=8, pady=5)

        return entry

    def _create_tabs(self):
        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True)

        self.axial_tab = ttk.Frame(tabs)
        self.radial_tab = ttk.Frame(tabs)

        tabs.add(self.axial_tab, text="Axial Field")
        tabs.add(self.radial_tab, text="Radial Field")

        self._create_axial_tab()
        self._create_radial_tab()

    def _create_axial_tab(self):
        controls = ttk.LabelFrame(
            self.axial_tab,
            text="Axial coil currents",
        )
        controls.pack(fill="x", padx=10, pady=10)

        self.injection_entry = self._entry(
            controls,
            "Injection current (A):",
            188.35,
            0,
        )
        self.center_entry = self._entry(
            controls,
            "Center current (A):",
            -155.18,
            1,
        )
        self.extraction_entry = self._entry(
            controls,
            "Extraction current (A):",
            135.74,
            2,
        )

        ttk.Button(
            controls,
            text="Calculate axial field",
            command=self._calculate_axial,
        ).grid(row=3, column=0, pady=10)

        self.axial_save_button = ttk.Button(
            controls,
            text="Save plot as image...",
            command=self._save_axial_plot,
            state="disabled",
        )
        self.axial_save_button.grid(row=3, column=1, pady=10)

        self.axial_status = ttk.Label(self.axial_tab, text="")
        self.axial_status.pack()

        self.axial_figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.axial_axes = self.axial_figure.subplots(2, 1)

        self.axial_canvas = FigureCanvasTkAgg(
            self.axial_figure,
            master=self.axial_tab,
        )
        self.axial_canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10,
        )

    def _create_radial_tab(self):
        controls = ttk.LabelFrame(
            self.radial_tab,
            text="Radial magnet current",
        )
        controls.pack(fill="x", padx=10, pady=10)

        self.sextupole_entry = self._entry(
            controls,
            "Sextupole current (A):",
            430.0,
            0,
        )

        ttk.Button(
            controls,
            text="Calculate radial field",
            command=self._calculate_radial,
        ).grid(row=1, column=0, columnspan=2, pady=10)

        self.radial_status = ttk.Label(self.radial_tab, text="")
        self.radial_status.pack()

        self.radial_figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.radial_axis = self.radial_figure.subplots()

        self.radial_canvas = FigureCanvasTkAgg(
            self.radial_figure,
            master=self.radial_tab,
        )
        self.radial_canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10,
        )
    def _draw_total_axial(
        self, ax, total, injection_index, extraction_index, minimum_index
    ):
        ax.clear()
        ax.plot(self.calculator.z_cm, total, linewidth=2)
        ax.axhline(
            0.64, color="red", linestyle="--",
            label="18 GHz resonance: 0.64 T",
        )
        ax.axhline(
            1.00, color="green", linestyle="--",
            label="28 GHz resonance: 1.00 T",
        )

        ax.set_title("Axial Magnetic Field")
        ax.set_xlabel("Z (cm)")
        ax.set_ylabel("Bz (T)")
        ax.grid(True, axis="y")

        resonance_legend = ax.legend(loc="upper right")
        ax.add_artist(resonance_legend)

        value_handles = [
            Line2D([], [], color="none",
                   label=f"$B_{{inj}}$ = {total[injection_index]:.2f} T"),
            Line2D([], [], color="none",
                   label=f"$B_{{extr}}$ = {total[extraction_index]:.2f} T"),
            Line2D([], [], color="none",
                   label=f"$B_{{min}}$ = {total[minimum_index]:.2f} T"),
        ]
        ax.legend(
            handles=value_handles,
            loc="upper left",
            handlelength=0,
            handletextpad=0,
            fontsize=9,
        )

    def _calculate_axial(self):
        try:
            injection = float(self.injection_entry.get())
            center = float(self.center_entry.get())
            extraction = float(self.extraction_entry.get())

            result = self.calculator.calculate_axial(
                injection,
                center,
                extraction,
            )

            total = result["total_t"]

            injection_mask = (
                (self.calculator.z_cm >= -100)
                & (self.calculator.z_cm <= 0)
            )
            extraction_mask = (
                (self.calculator.z_cm >= 0)
                & (self.calculator.z_cm <= 100)
            )
            minimum_mask = (
                (self.calculator.z_cm >= -31)
                & (self.calculator.z_cm <= 38)
            )

            injection_indices = np.where(injection_mask)[0]
            extraction_indices = np.where(extraction_mask)[0]
            minimum_indices = np.where(minimum_mask)[0]

            injection_index = injection_indices[
                np.argmax(total[injection_indices])
            ]
            extraction_index = extraction_indices[
                np.argmax(total[extraction_indices])
            ]
            minimum_index = minimum_indices[
                np.argmin(total[minimum_indices])
            ]

            self._axial_plot_data = {
                "total": total,
                "injection_index": injection_index,
                "extraction_index": extraction_index,
                "minimum_index": minimum_index,
            }
            self._draw_total_axial(self.axial_axes[0], **self._axial_plot_data)

            self.axial_axes[1].clear()
            self.axial_axes[1].plot(
                self.calculator.z_cm,
                result["injection_t"],
                label="Injection",
            )
            self.axial_axes[1].plot(
                self.calculator.z_cm,
                result["center_t"],
                label="Center",
            )
            self.axial_axes[1].plot(
                self.calculator.z_cm,
                result["extraction_t"],
                label="Extraction",
            )
            self.axial_axes[1].plot(
                self.calculator.z_cm,
                result["sextupole_t"],
                label="Sextupole baseline",
            )

            self.axial_axes[1].set_title("Axial Field Components")
            self.axial_axes[1].set_xlabel("Z (cm)")
            self.axial_axes[1].set_ylabel("Bz (T)")
            self.axial_axes[1].grid(True, axis="y")


            self.axial_figure.tight_layout()
            self.axial_canvas.draw()
            self.axial_save_button.config(state="normal")

            self.axial_status.config(
                text=(
                    f"Injection peak: "
                    f"{total[injection_index]:.2f} T at "
                    f"z = {self.calculator.z_cm[injection_index]:.1f} cm   |   "
                    f"Extraction peak: "
                    f"{total[extraction_index]:.2f} T at "
                    f"z = {self.calculator.z_cm[extraction_index]:.1f} cm   |   "
                    f"Central minimum: "
                    f"{total[minimum_index]:.2f} T at "
                    f"z = {self.calculator.z_cm[minimum_index]:.1f} cm"
                )
            )

        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Enter valid numeric axial currents.",
            )
    def _save_axial_plot(self):
        if self._axial_plot_data is None:
            return

        path = filedialog.asksaveasfilename(
            title="Save axial field plot",
            defaultextension=".png",
            initialfile="venus_axial_field.png",
            filetypes=[
                ("PNG image", "*.png"),
                ("PDF document", "*.pdf"),
                ("SVG image", "*.svg"),
                ("JPEG image", "*.jpg"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        # 9 x 6 inches = 3:2; at 250 dpi this is 2250 x 1500 pixels.
        export_figure = Figure(figsize=(6, 4))
        export_axis = export_figure.add_subplot()
        self._draw_total_axial(export_axis, **self._axial_plot_data)
        export_figure.tight_layout()

        try:
            export_figure.savefig(path, dpi=300)
        except Exception as error:
            messagebox.showerror(
                "Save failed",
                f"Could not save the image:\n{error}",
            )
            return

        self.axial_status.config(
            text=f"{self.axial_status.cget('text')}\nSaved to {path}"
        )

    def _calculate_radial(self):
        try:
            sextupole_current = float(self.sextupole_entry.get())
            field = self.calculator.calculate_radial(sextupole_current)

            self.radial_axis.clear()
            self.radial_axis.plot(
                RADIAL_RADIUS_CM,
                field,
                linewidth=2,
                label=f"Sextupole: {sextupole_current:g} A",
            )
            self.radial_axis.set_title("Radial Magnetic Field")
            self.radial_axis.set_xlabel("Radius (cm)")
            self.radial_axis.set_ylabel("Br (T)")
            self.radial_axis.grid(True, axis="y")
            self.radial_axis.legend()

            self.radial_figure.tight_layout()
            self.radial_canvas.draw()

            self.radial_status.config(
                text=(
                    f"Field at chamber wall (r = 7.2 cm): "
                    f"{field[-1]:.6f} T"
                )
            )

        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Enter a valid numeric sextupole current.",
            )


if __name__ == "__main__":
    app = VenusApp()
    app.mainloop()