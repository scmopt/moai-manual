---
title: Metrovi
marimo-version: 0.10.12
width: full
---

# 配送計画システムMETRO VI

> 配送計画システム METRO とその使用法



ここで考えるのは，ほとんどの実際問題を解けるようにするために，以下の一般化をした配送計画モデルである．

- 複数時間枠制約
- 多次元容量非等質運搬車
- 配達・集荷
- 積み込み・積み降ろし
- 複数休憩条件
- スキル条件
- 優先度付き
- パス型許容
- 複数デポ（運搬車ごとの発地，着地）


### Jobクラス

- id: ジョブを区別するための整数値
- location: 経度と緯度（浮動小数点数）から構成されるリスト．地図データを用いる場合には，これらの値を用いて移動時間を計算する．
- location_index: 移動時間データを用いる場合の地点に対応する番号． 地図データを用いる場合には使用しない．
- setup: 作業の準備時間（以下では時間の単位は全て秒であり，整数値をとるものとする．）
- service: 作業時間
- delivery： 配達量を表す整数値のリスト；荷量は複数の属性をもつ場合があるので，順序型として表現する．たとえば，重量，容積，パレット数にそれぞれ上限がある場合には，3次元のリスト（ベクトル）として入力する．
- pickup: 集荷量を表す整数値のリスト
- skills: ジョブを遂行するために必要なスキルを表す整数のリスト．少なくとも1つのスキルを定義する必要がある． 運搬車がジョブを処理可能か否かを判定するときに用いられる．
運搬車はジョブが要求するすべてのスキルをもたないと処理できない．
たとえば，4トン車と10トン車の2種類の運搬車があり，10トン車では入れない顧客がいる場合を考える．
10トン車では入庫不能な顧客（ジョブ）に対してはスキルを [0,1] と定義し， 入れる顧客に対しては [0] と定義する．
4トン車のスキルを [0,1] と，10トン車のスキルを [0] と定義すれば，10トン車はスキル1をもたないので，入庫不能な顧客を処理することができないことが表現できる．
- priority: ジョブの優先度を表す [0,100] の整数値．ジョブを処理しない（どの運搬車にも割り当てない）としたときに支払われるペナルティを表す．優先度が大きいジョブほど運搬車に割り当てられる（処理される）可能性が高くなる．
- time_windows： 時間枠（作業開始可能時刻，終了時刻の組）を表す長さ2のタプルの順序型．複数の時間枠を表すことができる．
たとえば，午前中と午後に作業が可能で，昼の時間帯には作業不能である顧客に対しては， [[8:00,12:00],[13:00,17:00] ]の時刻を基準時刻からの秒に換算したものを入力する．
すなわち8時を基準とした場合には，[[0,14400], [18000, 32400] ] と入力する．
- description: 名称などを文字列として入力する．

```{.python.marimo}
# | export
MAX_UINT = int(numpy.iinfo(numpy.uintp).max)
MAX_INT = int(numpy.iinfo(numpy.intp).max)


class Entity(BaseModel):
    def __str__(self) -> str:
        repr = {
            k: v
            for k, v in self.model_dump().items()
            if k in self.model_fields_set
        }
        return str(repr)


class Job(Entity):
    id: int
    location: Union[None, Sequence[float]] = (
        None  # 経度・緯度 [lon, lat] を入れる場合
    )
    location_index: Union[None, int] = None  # 行列のインデックスを入れる場合
    setup: Optional[int] = 0
    service: Optional[int] = 0
    delivery: Optional[Sequence[int]] = [0]
    pickup: Optional[Sequence[int]] = [0]
    skills: Optional[Set[int]] = None
    priority: Optional[int] = 0
    time_windows: Optional[Sequence[Tuple[int, int]]] = None
    description: Optional[str] = ""
```

```{.python.marimo}
_job1 = Job(id=1414, location_index=0)
_job1.model_dump_json(exclude_none=True)
```

### ShipmentStepクラス

- id: ジョブを区別するための整数値
- location: 経度と緯度（浮動小数点数）から構成されるリスト．地図データを用いる場合には，これらの値を用いて移動時間を計算する．
- location_index: 移動時間データを用いる場合の地点に対応する番号． 地図データを用いる場合には使用しない．
- setup: 作業の準備時間（以下では時間の単位は全て秒であり，整数値をとるものとする．）
- service: 作業時間
- time_windows： 時間枠（作業開始可能時刻，終了時刻の組）を表す長さ2のタプルの順序型．複数の時間枠を表すことができる．
- description: 名称などを文字列として入力する．
<!---->
### Shipmentクラス

- pickup: 積み込み地点のShipmentStep
- delivery_point: 積み降ろし地点のShipmentStep
- amount: 積み込み地点で積み込み，積み降ろし地点で降ろす量（輸送量）．整数値の順序型．
- skills: 輸送を行うために必要なスキル
- priority: 優先度

```{.python.marimo}
# | export
class ShipmentStep(Entity):
    id: int
    location: Optional[Tuple[float, float]] = None
    location_index: Optional[int] = None
    setup: Optional[int] = 0
    service: Optional[int] = 0
    time_windows: Optional[Sequence[Tuple[int, int]]] = None
    description: Optional[str] = ""


class Shipment(Entity):
    pickup: ShipmentStep
    delivery: ShipmentStep
    amount: Optional[Sequence[int]] = [0]
    skills: Optional[Set[int]] = None
    priority: Optional[int] = 0
```

### Breakクラス

- id: 休憩を区別するための整数値
- time_windows: 休憩をとる時間枠（最早時刻と最遅時刻のタプル）の順序型
- service: 休憩時間
- description: 休憩の説明
- max_load: 休憩をとることが可能な最大積載量を表す整数の順序型．休憩をとる時間帯ごとに，最大積載量が指定した値より大きい運搬車の休憩を禁止するために使われる．
  たとえば，危険物を一定値よりたくさん積んでいる場合には危険なので，休憩所には入れないことを表す．

```{.python.marimo}
# | export
class Break(Entity):
    id: int
    time_windows: Optional[Sequence[Tuple[int, int]]] = []
    service: Optional[int] = 0
    description: Optional[str] = ""
    max_load: Optional[Sequence[int]] = None
```

```{.python.marimo}
for id, row in enumerate(break_df.itertuples()):
    # print(ast.literal_eval(row.time_windows))
    _temp = {
        "id": id,
        "description": row.description,
        "time_windows": ast.literal_eval(row.time_windows),
        "service": row.service,
    }
    _b = Break(**_temp)
_b
```

### Vehicleクラス

- id: 運搬車を区別するための整数値
- start: 運搬車の出発地点を表す経度・緯度のタプル．(start_indexで点の番号を参照させ，移動時間行列を与える方法も可能である．） 省略した場合には，最初の訪問地点から出発する．
- end: 運搬車の最終到着地点を表す経度・緯度の順序型． 省略した場合には，最後の訪問地点で終了する． ただし，出発地点と最終到着地点の両者を省略することはできない． 出発地点と同じ座標にした場合には，出発地点であるデポに戻ることを表す．
- start_index: 移動時間データを用いる場合の運搬車の出発地点に対応する番号． 地図データを用いる場合には使用しない． このデータが空 (NaN) の場合には，出発地点を指定せず，最初に訪問した地点から運搬車の経路が開始される．
- end_index: 移動時間データを用いる場合の運搬車の最終到着地点に対応する番号． 地図データを用いる場合には使用しない． このデータが空 (NaN) の場合には，最終到着地点を指定せず，最後に訪問した地点で運搬車の経路が終了する．
もちろん出発地点と最終到着地点が異なっても良い．そのため，複数デポや最終地点が車庫であることも表現できる．
- profile: 運搬車の種類（プロファイル）を表す文字列．既定値は "car" だが， "bicycle", "walk", "truck" などと指定すると， 該当する移動時間や費用を使うことができる．これによって運搬車ごとの移動時間や費用の違いを表現できる．
- capacity: 運搬車の積載量上限（容量）を表す整数値のリスト．ジョブデータの集荷量・配達量，輸送データの輸送量と同じ長さの順序型である必要がある．
- time_window: 運搬車の時間枠（最早開始時刻と最遅到着時刻の組）を表す長さ2のタプル
- skills: 運搬車のもつスキル． ジョブや輸送で要求するすべてのスキルをもたないと処理できない．
- breaks: 休憩を表すインスタンスの順序型
- description: 名称などを文字列として入力する．
- costs: VehicleCostsのインスタンス（固定費用 fixed，1時間あたりの費用 per_hour，1kmあたりの費用 per_km を設定できる．） これによって，運搬車ごとに異なる費用を，簡単に表現できる．
- speed_factor: 既定値の速度の何倍かを表す浮動小数点数．既定値は $1.0$． これを使うと，運搬車ごとに異なる移動時間をもつことを，簡易的に表現することができる．
- max_tasks: 処理可能なジョブ数の上限
- max_travel_time: 最大稼働時間

```{.python.marimo}
# | export
class VehicleCosts(Entity):
    fixed: Optional[int] = 0
    per_hour: int = 3600
    per_km: Optional[int] = 0


class VEHICLE_STEP_TYPE(Entity):
    START: str = "start"
    END: str = "end"
    BREAK: str = "break"
    SINGLE: str = "single"
    PICKUP: str = "pickup"
    DELIVERY: str = "delivery"


class VehicleStep(Entity):
    step_type: VEHICLE_STEP_TYPE
    id: Optional[int] = None
    service_at: Optional[int] = None
    service_after: Optional[int] = None
    service_before: Optional[int] = None


class Vehicle(Entity):
    id: int
    start: Union[None, Sequence[float]] = None
    end: Union[None, Sequence[float]] = None
    start_index: Union[None, int] = None
    end_index: Union[None, int] = None
    profile: Optional[str] = "car"
    capacity: Optional[Union[Sequence[int]]] = None
    skills: Optional[Set[int]] = None
    time_window: Optional[Tuple[int, int]] = None
    breaks: Optional[Sequence[Break]] = None
    description: str = ""
    costs: VehicleCosts = VehicleCosts()
    speed_factor: Optional[float] = 1.0
    max_tasks: Optional[int] = None
    max_travel_time: Optional[int] = None
    steps: Sequence[VehicleStep] = None
```

```{.python.marimo}
_vehicle = Vehicle(
    id=7,
    start_index=0,
    end_index=0,
    time_window=(0, 100),
    costs=VehicleCosts(per_hour=10),
)
# eval(repr(vehicle))
print(_vehicle.model_dump_json(exclude_none=True))
```

### Matrixクラス

- durations: 移動時間を表す行列（リストのリスト）
- distances: 移動距離を表す行列（リストのリスト）
- costs: 移動費用を表す行列（リストのリスト）；costsが定義されていない場合には，Vehicleクラスで定義される運搬車ごとの費用（固定，1時間あたり，1kmあたり）から計算される．

```{.python.marimo}
# | export
class Matrix(Entity):
    durations: List[List] = None
    distances: List[List] = None
    costs: List[List] = None
```

### Modelクラス

配送計画モデルを表すクラス

- jobs: Jobインスタンスの順序型
- shipments: Shipmentインスタンスの順序型
- vehicles: Vehicleクラスの順序型
- matrices: 運搬車のプロファイルを表す文字列をキーとし，Matrixクラスのインスタンスを値とした辞書

```{.python.marimo}
# | export
class Model(Entity):
    jobs: Optional[Sequence[Job]] = Field(
        description="ジョブたち", default=None
    )
    shipments: Optional[Sequence[Shipment]] = Field(
        description="輸送たち", default=None
    )
    vehicles: Optional[Sequence[Vehicle]] = Field(
        description="運搬車たち", default=None
    )
    matrices: Optional[Dict[str, Matrix]] = Field(
        description="行列たち", default=None
    )
```

### 小さなモデルを用いたテスト

```{.python.marimo}
_break0 = Break(id=0, time_windows=[(1000, 2000)], service=100, max_load=[20])

_model = Model()
_model.vehicles = [
    Vehicle(
        id=7,
        start_index=0,
        end_index=0,
        capacity=[100],
        time_window=[0, 20000],
        costs=VehicleCosts(fixed=99999),
        breaks=[_break0],
    ),
    Vehicle(
        id=8, start_index=2, end_index=2, capacity=[100], breaks=[_break0]
    ),
]
_model.jobs = [
    Job(
        id=1,
        location_index=0,
        delivery=[10],
        pickup=[30],
        time_windows=[(0, 100), (300, 10000)],
    ),
    Job(id=12, location_index=1, delivery=[20]),
    Job(id=1616, location_index=2, delivery=[30]),
    Job(id=1717, location_index=3, delivery=[40]),
]
_model.matrices = {
    "car": Matrix(
        durations=[
            [0, 2104, 197, 1299],
            [2103, 0, 2255, 3152],
            [197, 2256, 0, 1102],
            [1299, 3153, 1102, 0],
        ]
    )
}
_model.shipments = [
    Shipment(
        pickup=ShipmentStep(id=1, location_index=2),
        delivery=ShipmentStep(
            id=124, location_index=3, time_windows=[(500, 2000)]
        ),
        amount=[10],
    )
]
pprint(_model.model_dump_json(exclude_none=True))

_input_dic, _output_dic, _error = optimize_vrp(
    _model, matrix=True, explore=5, cloud=False, osrm=False, host=host
)

_summary_df, _route_summary_df, _unassigned_df, _route_df_dic = make_solution(
    _output_dic
)
_summary_df
```

## 最適化関数 optimize_vrp

引数：

- model: 最適化モデルを入れた辞書
- matrix = False: 移動時間データを用いる場合True
- thread: 最適化に使用するスレッド数
- explore: 探索の度合いを表すパラメータ； 0から5の整数で，大きいほど念入りに探索する．
- cloud: 複数人が同時実行する可能性があるときTrue（既定値はFalse）; Trueのとき，ソルバー呼び出し時に生成されるファイルにタイムスタンプを追加し，計算終了後にファイルを消去する．
- osrm: OSRMを外部のサーバーで呼び出しをしたいときTrue， localhostで呼び出すときFalse


返値：

- input_dic: データ入力のためのJSONデータ
- output_dic: 結果出力のためのJSONデータ
- error: エラーメッセージを入れた文字列

```{.python.marimo}
def optimize_vrp(
    model,
    matrix=False,
    threads=4,
    explore=5,
    cloud=False,
    osrm=False,
    host="localhost",
):
    if cloud:
        time_stamp = dt.datetime.now().timestamp()
    else:
        time_stamp = 1
    with open(f"test{time_stamp}.json", "w") as f:
        f.write(model.model_dump_json(exclude_none=True))
    if cloud:
        import pathlib

        p = pathlib.Path(".")
        script = p / "scripts/metroVI"
    else:
        script = "./metroVI"
    if platform.system() == "Windows":
        if matrix:
            cmd = f"metro-win -i test{time_stamp}.json -o output{time_stamp}.json"
        else:
            cmd = f"metro-win - g -i test{time_stamp}.json -o output{time_stamp}.json"
    elif platform.system() == "Darwin":
        if platform.processor()[0] == "i":
            if matrix:
                cmd = f"{script}-mac-intel -i test{time_stamp}.json -o output{time_stamp}.json"
            elif osrm:
                cmd = f"{script}-mac-intel -g -i test{time_stamp}.json -o output{time_stamp}.json -a car:{host}"
            else:
                cmd = f"{script}-mac-intel -g -i test{time_stamp}.json -o output{time_stamp}.json"
        elif platform.processor()[0] == "a":
            if matrix:
                cmd = f"{script}-mac-sillicon -i test{time_stamp}.json -o output{time_stamp}.json"
            elif osrm:
                cmd = f"{script}-mac-sillicon -g -i test{time_stamp}.json -o output{time_stamp}.json -a car:{host}"
            else:
                cmd = f"{script}-mac-sillicon -g -i test{time_stamp}.json -o output{time_stamp}.json"
        else:
            print(
                f"{platform.system()} and {platform.processor()} may not be supported."
            )
            return -1
    elif platform.system() == "Linux":
        if matrix:
            cmd = f"{script}-linux-intel -i test{time_stamp}.json -o output{time_stamp}.json"
        elif osrm:
            cmd = f"{script}-linux-intel -g -i test{time_stamp}.json -o output{time_stamp}.json -a car:{host}"
        else:
            cmd = f"{script}-linux-intel -g -i test{time_stamp}.json -o output{time_stamp}.json"
    else:
        print(platform.system(), "may not be supported.")
        return -1
    cmd = cmd + (" -t " + str(threads) + " -x " + str(explore))
    try:
        print("Now solving ...")
        o = subprocess.run(cmd.split(), check=True, capture_output=True)
        print("Done")
    except subprocess.CalledProcessError as e:
        return ("", "", e.stderr)
    example1_in = open(f"test{time_stamp}.json", "r")
    input_dic = json.load(example1_in)
    example1_out = open(f"output{time_stamp}.json", "r")
    output_dic = json.load(example1_out)
    try:
        error = output_dic["error"]
    except:
        error = ""
    if cloud:
        os.remove(p / f"test{time_stamp}.json")
        os.remove(p / f"output{time_stamp}.json")
    return (input_dic, output_dic, error)
```

### Solutionクラス

```{.python.marimo}
class Summary(Entity):
    cost: int
    routes: int
    unassigned: int
    setup: int
    service: int
    duration: int
    waiting_time: int
    priority: int
    violations: Optional[Sequence[str]]
    delivery: Optional[Sequence[int]]
    pickup: Optional[Sequence[int]]
    distance: Optional[int] = None


class Violation(Entity):
    cause: str
    duration: Optional[int] = None


class Route_1(Entity):
    vehicle: int
    steps: Optional[Sequence[Any]]
    cost: int
    setup: int
    service: int
    duration: int
    waiting_time: int
    priority: int
    violations: Optional[Sequence[Violation]] = None
    delivery: Optional[Sequence[int]] = None
    pickup: Optional[Sequence[int]] = None
    description: str = None
    geometry: Optional[str] = None
    distance: Optional[int] = None


class Solution_1(Entity):
    code: int
    error: Optional[str] = None
    summary: Optional[Summary] = None
    unassigned: Optional[Sequence[Any]] = None
    routes: Optional[Sequence[Route_1]] = None
```

### 解の情報を計算する関数 make_solution

引数：

- output_dic: 最適化の結果を格納した辞書

返値：

- summary_df: 解の概要のデータフレーム
- route_summary_df: ルートの概要のデータフレーム
- unassigned_df: 未割り当てのジョブ・輸送のデータフレーム
- route_df_dic: キーをルート番号とし，ルートの情報を格納したデータフレームを値とした辞書．
  ルートデータフレームの列は以下の通り．

     - index: 訪問順序
     - type: 地点名（開始地点はstart，終了地点はend，ジョブはjob，輸送の積み込み地点はpickup，積み降ろし地点はdelivery，休憩はbreakと表示される．）
     - cost: ルートの費用
     - location: 経度・緯度
     - location_index: 地点インデックス
     - setup: 準備時間
     - service: 作業時間
     - waiting_time: 待ち時間
     - load: その地点での積載量を表すリスト
     - arrival: 到着時刻
     - duration: 累積移動時間
     - priority: 優先度の合計
     - violations: 制約逸脱量
     - delivery: 配達量の合計
     - pickup: 積み込み量の合計
     - description: 概要
     - geometry: ルート詳細のpolyline
     - distance: 総走行距離
     - id: ジョブ・輸送の番号
     - job: ジョブ

```{.python.marimo}
def make_solution(output_dic: dict) -> tuple:
    solution = Solution_1.model_validate(output_dic)
    sol_dict = solution.model_dump()
    summary_df = pd.DataFrame.from_dict(sol_dict["summary"], orient="index").T
    if len(sol_dict["routes"]) > 0:
        dfs = []
        for r in sol_dict["routes"]:
            df = pd.DataFrame.from_dict(r, orient="index").T
            df.drop("steps", axis=1, inplace=True)
            dfs.append(df)
        route_summary_df = pd.concat(dfs)
        route_summary_df.reset_index(inplace=True)
        route_summary_df.drop("index", axis=1, inplace=True)
    else:
        route_summary_df = None
    if len(sol_dict["unassigned"]) > 0:
        dfs = []
        for r in sol_dict["unassigned"]:
            df = pd.DataFrame.from_dict(r, orient="index").T
            dfs.append(df)
        unassigned_df = pd.concat(dfs)
        unassigned_df.reset_index(inplace=True)
        unassigned_df.drop("index", axis=1, inplace=True)
    else:
        unassigned_df = None
    route_df_dic = {}
    for r in sol_dict["routes"]:
        if len(sol_dict["routes"]) > 0:
            dfs = []
            for j in r["steps"]:
                df = pd.DataFrame.from_dict(j, orient="index").T
                dfs.append(df)
            route_df = pd.concat(dfs)
            route_df.reset_index(inplace=True)
            route_df.drop("index", axis=1, inplace=True)
            route_df_dic[r["vehicle"]] = route_df
    return (summary_df, route_summary_df, unassigned_df, route_df_dic)
```
