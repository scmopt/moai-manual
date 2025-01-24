---
title: Scml2
marimo-version: 0.10.12
width: full
---

# SCML (Supply Chain Modeling Language)

> Supply Chain Modeling Language Class

```{.python.marimo}
# | default_exp scml2
```

```{.python.marimo}
# | export

# from scmopt2.core import SCMGraph

import random
from pprint import pprint

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from typing import List, Optional, Union, Tuple, Dict, Set, Any, DefaultDict
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    validator,
    confloat,
    conint,
    constr,
    Json,
)
from pydantic.tools import parse_obj_as
from datetime import datetime, date, time

import networkx as nx
import copy
import pickle
from collections import defaultdict
import pandas as pd
import numpy as np
import scipy

import graphviz
import matplotlib.pyplot as plt

# ソルバー GurobipyのときにはTrueにする（Gurobiは model.updateが必要なため）
GUROBI = False

if GUROBI == False:
    # import mindoptpy as gp
    # gp.GRB = gp.MDO
    import mypulp as gp
    # import pyscipopt as gp

folder = "./data/"
# from IPython.display import Image, YouTubeVideo
# import re
# import requests
# from collections import OrderedDict, defaultdict
# import subprocess
# import platform
# import json
# import numpy as np
# import plotly

# import ast  #文字列からオブジェクトを起こすモジュール

# from openpyxl import Workbook, load_workbook
# from openpyxl.worksheet.table import Table, TableStyleInfo
# from openpyxl.chart import ScatterChart, Reference, Series
# from openpyxl.worksheet.datavalidation import DataValidation
# from openpyxl.formatting.rule import ColorScaleRule, CellIsRule, FormulaRule
# from openpyxl.styles import Color, PatternFill, Font, Border, Alignment
# from openpyxl.styles.borders import Border, Side
# from openpyxl.utils.dataframe import dataframe_to_rows
# from openpyxl.comments import Comment
```

## SCMLとは

ここで提案する SCML (Suply Chain Modeling Language) とは， 様々なサプライ・チェイン最適化モデルを
統一的に表現するための基本言語である．

サプライ・チェインを対象とした最適化モデルには，様々なものがある．
同じ現実問題を対象としていても，その抽象化のレベルによって，様々なモデルが作成できる．
エンドユーザーが使用することを想定した場合には，
実ロジスティクス・オブジェクト（トラックや機械などを指す造語）を用いたモデルが有効であり，
一方，システムを作成するロジスティクス・エンジニアや研究者が使用することを想定した場合には，
抽象ロジスティクス・オブジェクト（資源やネットワークなどを指す造語）を用いたモデルが有効である．

また，同じ抽象モデルでも，グラフ・ネットワークなど高度に抽象化されたものから，
時間枠付き配送計画や資源制約付きスケジューリングなど具体的な応用を対象としたものまで
様々な階層に分けることができ，それぞれ利点・弱点がある．

抽象度の階層の最下層にあるのが，実際の現場で用いられるシステムに内在するモデルであり，
それらは現実モデルに直結している．最下層の現実モデルの上位にあるのが，様々な典型的なサプライ・チェイン最適化モデル
であり，個々のモデルに対して多くの研究がなされている．
この階層のモデルの例として，ロジスティクス・ネットワーク設計モデル，安全在庫配置モデル，スケジューリングモデル，配送計画モデルなどがあげられる．
ここで考えるSCMLは，これらのモデルを統一的に表現するために設計されている．

サプライ・チェインの理論や実務については，様々な場所で分野を超えた交流が成されているが，
分野ごとに使用されている用語の意味や定義が異なるため，議論が噛み合わず，有益な交流ができているとは言い難い．
たとえば，ある分野ではサプライ・チェイン最適化とは「在庫」を適正化するための手法として捉えられており，
別の分野ではサプライ・チェイン最適化とは工場内のスケジューリングを指していたりする．

異分野間の交流は，サプライ・チェインのような複合的な学問体系にとっては必要不可欠なものであるが，
これらの現象を目のあたりにし，研究者と実務家が同じ土俵で議論するためのモデルの必要性を痛切に感じた．
これが，SCMLを考えるに至った動機である．

ここで提案する抽象モデルでは，サプライ・チェインを，
空間（点と枝からなるネットワーク），
時間（期），製品，資源，活動（とそれを実行する方法であるモード）などの基本構成要素
から構成されるものと捉える．
構成要素の中心に活動があり，
活動（とそれに付随するモード）が資源を用いて製品を時・空間内に移動させることが
サプライ・チェインの本質であり，目的であると考える．

以下では，これらの基本構成要素 (entity) をPython/Pydanticを用いたクラスとして表現していく．
また，これらを組み込んだモデルクラスを設計し，最適化や可視化のためのメソッドを準備する．
<!---->
## Basic Entity Class

準備のために基本となるEntityクラスを作っておく。PydanticのBaseModelから派生させる。

Entity間に，集約（グループ化）と非集約関係を定義するために，集約したEntityを表す親 (parent) と，非集約したときのEntityのリストを表す子リスト (children) を定義しておく。
たとえば，月が親のとき子はその月に含まれる週や日になる。

属性

- name: 名称（整数か文字列）
- parent: 親Entityの名称
- children: 子Entityの名称のリスト


親子関係の定義には，makeGroup関数などを用いる。

```{.python.marimo}
# | export
class Entity(BaseModel):
    """ """

    name: Union[int, str] = Field(description="名称")
    parent: Optional[Union[int, str]] = Field(
        description="親の名称", default=None
    )
    children: Optional[List[Union[int, str]]] = Field(
        description="子の名称のリスト", default=None
    )

    def to_pickle(self, file_name=None):
        if file_name is None:
            file_name = str(self.name)
        with open(f"{file_name}.pkl", "wb") as f:
            pickle.dump(self, f)

    def setParent(self, parent: Union[int, str]) -> None:
        self.parent = parent.name
        if parent.children is None:
            parent.children = []
        parent.children.append(self.name)

    def addChildren(self, children: List[Union[int, str]]) -> None:
        if self.children is None:
            self.children = []

        for child in children[:]:
            self.children.append(child.name)
            child.parent = self.name
```

### 使用例

TODO: 親子関係をmermaidで描画

```{.python.marimo}
_entity1 = Entity(name="entity1")
_entity2 = Entity(name="entity2")
_entity3 = Entity(name="entity3")

_entity1.setParent(_entity2)
print(_entity2)

_entity1.addChildren([_entity2, _entity3])
print(_entity1)
```

## Period

期 (Period) は，時間をモデル化するために用いられる連続時間を離散化したものである．
Entityクラスから派生させるので，name, parent, chidren属性をもつ（以下同様）。

最も単純な期集合は，有限な正数 $T$，時刻の列 $0= t_1 < t_2 < \cdots < t_T$ を与えたとき，
区間 $(t_i,t_{i+1}] ~(i=1,\cdots,T-1)$ の順序付き集合として生成される．
$t_{i+1}-t_i$ がすべて同じであるとき，$t_{i+1}-t_i (=\delta)$ を期集合の幅とよぶ．
サプライ・チェインをモデル化する際には，
意思決定レベルの違いにより様々な幅をもつ期集合が定義される．
ここでは，それらを集めたものを「期」と定義する．
期（集合）に対しても，点と同様に集約・非集約関係が定義できる．
たとえば，日を集約したものが週であり，週を集約したものが月（もしくは四半期や年）となる．

期は，名前 name と開始時刻を表す start を引数として生成される．
startは整数，浮動小数点数，日付時刻型，日付型，時刻型のいずれかの型をもつ．
モデルに追加された期は非減少順（同じ場合には名前の順）に並べ替えられ，
小さい順に $0,1,2,\cdots$ とインデックスが付けられる．

モデルの最適化の際には，開始期と終了期を指定して，一部の期間に対してだけ最適化することができる．

属性

- start: 開始時刻
<!-- - end: 終了時刻 -->

Modelに付加された期インスタンスは，startの非減少順（同点の場合には名前順）に並べられ，$0$ から始まる整数値のインデックスが付与される．
最適化は，モデルに与えられた最適化期間（省略された場合はすべての期間）に対して行われる．

```{.python.marimo}
# | export
class Period(Entity):
    start: Optional[Union[int, float, datetime, date, time]] = Field(
        description="開始時刻", dafault=None
    )
    # end: Optional[ Union[datetime,date,time] ]      = Field( description="終了時刻", default = None)
```

### 使用例

```{.python.marimo}
_period1 = Period(name="Period1", start="2023-04-23T10:20:30.400+02:30")
_period1
```

## Resource

Resourceは，有限な資源を表すクラスである。
サプライ・チェインを構成する企業体は，
生産ライン，機械，輸送機器（トラック，船，鉄道，飛行機），
金（財務資源），人（人的資源）などの資源から構成される．
資源集合に対しても，点と同様に集約・非集約関係が定義できる．

属性

- rtype: 資源のタイプを表す文字列もしくはNone（'break', 'max','vehicle'などで，既定値は None）
  - None: 通常の作業
  - break: 休憩可能な活動の場合に用いる休憩中の資源使用量
  - max: 並列実行可能な活動の場合に用いる並列実行中の最大資源使用量
  - vehicle: 移動可能な資源（運搬車）
- capacity: 資源量上限（容量）を表す整数，浮動小数点数，もしくは期ごとの資源量上限を保持する辞書
- fixed_cost: 資源を使用する場合にかかる固定費用
- modes: 資源を使用するモード名をキーとし，値に固定量と変動量のタプルを入れた辞書（モードに資源を追加する際に自動的に更新される。）

メソッド

- addCapacity(Period, amount): 期に容量amountを追加する。

```{.python.marimo}
# | export
class Resource(Entity):
    rtype: Optional[str] = Field(description="資源のタイプ", default=None)
    capacity: Optional[Union[int, float, Dict]] = Field(
        description="容量", default=None
    )
    fixed_cost: Optional[Union[int, float]] = Field(
        description="資源固定費用", default=0
    )
    modes: Optional[Dict[str, Tuple]] = Field(
        description="資源を使用するモード名をキーとし，固定量と変動量のタプルを入れた辞書",
        default=None,
    )

    def addCapacity(self, period: Period, amount: Union[int, float]) -> None:
        """
        Adds a capacity to the resource.
        """
        if self.capacity is None:
            self.capacity = {}
        # data = copy.deepcopy(self.capacity)
        # data.update({period.name: amount})
        # self.capacity = data
        self.capacity[period.name] = amount
```

### 使用例

```{.python.marimo}
_period1 = Period(name="Period1", start="2023-04-23T10:20:30.400+02:30")
_res1 = Resource(name="Res1")
_res1.addCapacity(period=_period1, amount=10)
_res1
```

## Product

Productは，製品（品目，半製品，原材料，部品，中間製品，完成品など）を表すクラスである．
製品は，外部から供給され，系内を姿形を変えながら移動した後，外部へ出て行く「もの」
である．製品は，ネットワークフローのモデルの用語では，品種（commodity）とよばれ，資源とは区別される．

属性

- weight: 重量（資源使用量に抽象化しているので，必要なし？）
- volume: 容積
- value: 製品の価値；在庫費用の計算に用いる。（モデルのデータに期別，地点別で保管するので必要なし？）

```{.python.marimo}
# | export
class Product(Entity):
    weight: Optional[float] = Field(description="重量", default=0.0)
    volume: Optional[float] = Field(description="容量", default=0.0)
    value: Optional[float] = Field(
        description="価値（点に依存しない場合）", default=0.0
    )
    # モデルクラスに移動
    # value: Optional[Dict[str, float]] = Field(description="点上での製品の価値（点の名称をキー，値を価値とした辞書）")
    # demand, supplyも同様に定義
    # def addValue(self, node_name:str, value: float) -> None:
    #     if self.value is None:
    #         self.value = {}
    #     self.value[node_name] = value

    # 在庫も地点ごとに入力?
    # safety_inventory: Optional[float] = Field(description="安全在庫量（最終期の目標在庫量）")
    # initial_inventory: Optional[float] = Field(description="初期在庫量")
    # target_inventory: Optional[float] = Field(description="目標在庫量")
    class Config:
        json_schema_extra = {
            "example": {
                "name": "いちご大福",
                "weight": 0.2,
                "value": 100,
            }
        }
```

### 使用例

```{.python.marimo}
_prod1 = Product(name="Prod1")
_prod1.to_pickle("sample1")
with open(f"sample1.pkl", "rb") as _f:
    _prod2 = pickle.load(_f)
_prod2
```

## Mode

Mode（モード，方策）は，以下でt定義する Activityクラスに付随し，活動を行うための具体的な方法を表す．
方策 (policy) は動的最適化の枠組みにおいては，システムの状態 (state) を行動 (action) への写像として定義されるが，
ここでは活動 (activity) を実行するための方法として捉えるものとする．
また，製品がどの原材料や部品から構成されるかを表す部品展開表 (Bill Of Materials: BOM) の情報ももつ。

属性

- components: 部品（原材料）名称をキー，必要数量を値とした辞書
- byproducts: 副生成物名称をキー，生成数量を値とした辞書
- fixed_requirement: 活動を行う際に必要な資源の固定量（段取り時間など）を保管した辞書
- variable_requirement: 活動を行う際に必要な資源の変動量（1単位製造するための作業時間など）を保管した辞書
- fixed_cost: 活動を行う際に必要な固定費用
- variable_cost: 活動量によって変化する変動費用の係数
- piecewise_x：活動量に対する費用を区分的線形関数で表す場合の $x$ 座標のリスト
- piecewise_y: 活動量に対する費用を区分的線形関数で表す場合の $y$ 座標のリスト
- upper_bound: 活動量上限
- duration: 作業時間（安全在庫配置モデルで使用）
- service_time: サービス時間（保証リード時間）； 下流の在庫地点が注文をしてから在庫が補充されるまでの時間の上限（安全在庫配置モデルで使用）
- service_level: サービス水準； $0$ 以上， $1$ 以下の浮動小数点数 （品切れが起きない確率；在庫モデルで使用）

関連付けられる活動のタイプが生産(make)の場合には，生産量を表す変数 $x_{m}^p$ が定義され，
段取り(setup)の場合には，生産量 $x_{m}^p$ と段取りの有無を表す $0$-$1$ 変数 $y_{m}^p$ が定義される．
ここで$p$ は活動に付随する製品である．

TODO: 一般化としての区分的線形な費用

- activities: モードが追加された活動の辞書（名前をキーとし，活動オブジェクトを値とする．）
- periods: シフト最適化で用いる期の名称をキー，期インスタンスを値とした辞書（シフトでカバーされる期の集合を保持する．）

メソッド

- addResource: 資源，固定量，変動量を与え，必要資源量を定義する。
- addCost: 費用を設定する。
- addComponent: 部品と必要量を定義する。
- addByproduct: 副生成物と生成量を定義する。
- addPeriod: シフトによってカバーされる期を追加する．

```{.python.marimo}
# | export
class Mode(Entity):
    components: Optional[Dict[str, float]] = Field(
        description="部品名称をキー，必要数量を値とした辞書", default=None
    )
    byproducts: Optional[Dict[str, float]] = Field(
        description="副生成物名称をキー，生成数量を値とした辞書", default=None
    )

    fixed_requirement: Optional[Dict[str, float]] = Field(
        description="資源名をキー，固定量とした辞書", default=None
    )
    variable_requirement: Optional[Dict[str, float]] = Field(
        description="資源名をキー，変動量とした辞書", default=None
    )

    fixed_cost: Optional[float] = Field(description="固定費用", default=0.0)
    variable_cost: Optional[float] = Field(description="変動費用", default=0.0)
    piecewise_x: Optional[List[float]] = Field(
        description="区分的線形費用関数のx座標のリスト", default=None
    )
    piecewise_y: Optional[List[float]] = Field(
        description="区分的線形費用関数のy座標のリスト", default=None
    )

    upper_bound: Optional[float] = Field(
        description="活動量上限", default=None
    )
    activities: Optional[set] = Field(
        description="モードを含む活動の集合", default=None
    )

    # シフト最適化用
    periods: Optional[Dict[Union[int, str], Period]] = Field(
        description="期名称をキー，期インスタンスとした辞書", default=None
    )

    # requirement: Optional[Dict[Tuple,Dict]]
    duration: Optional[Union[int, float]] = Field(
        description="作業時間", default=None
    )
    service_time: Optional[Union[int, float]] = Field(
        description="サービス（保証リード）時間", default=None
    )
    service_level: Optional[float] = Field(
        description="サービス水準", default=0.9
    )

    # breakable: Optional[Dict]                      = Field( description="分割指定を表す辞書" )
    # parallel:  Optional[Dict]                      = Field( description="並列実行指定を表す辞書" )
    # state:     Optional[Dict]                      = Field( description="状態推移を表す辞書" )
    # def addResource(self, resource:Resource, requirement:Union[float,int,Dict], rtype:Optional[str]=None) -> None:
    #     if self.requirement is None:
    #         self.requirement = {}
    #     if rtype is None:
    #         self.requirement[ (resource.name,None)] = requirement
    #     elif rtype in ["break", "max"]:
    #         self.requirement[ (resource.name,rtype)] = requirement
    #     else:
    #         raise TypeError("rtype must be None, break or max")
    def addResource(
        self, resource: Resource, fixed: float = 0.0, variable: float = 0.0
    ) -> None:
        if self.fixed_requirement is None:
            self.fixed_requirement = {}
        if self.variable_requirement is None:
            self.variable_requirement = {}
        self.fixed_requirement[resource.name] = fixed
        self.variable_requirement[resource.name] = variable
        if resource.modes is None:
            resource.modes = {}
        resource.modes[self.name] = (fixed, variable)

    def addCost(
        self,
        fixed: float = 0.0,
        variable: float = 0.0,
        piecewise_x: List[float] = None,
        piecewise_y: List[float] = None,
    ) -> None:
        self.fixed_cost = fixed
        self.variable_cost = variable
        self.piecewise_x = piecewise_x
        self.piecewise_y = piecewise_y

    def addComponent(self, component: Product, quantity: float = 1.0) -> None:
        if self.components is None:
            self.components = {}
        self.components[component.name] = quantity

    def addByproduct(self, byproduct: Product, quantity: float = 1.0) -> None:
        if self.byproducts is None:
            self.byproducts = {}
        self.byproducts[byproduct.name] = quantity

    def addPeriod(self, period: Period) -> None:
        if self.periods is None:
            self.periods = {}
        self.periods[period.name] = period
```

### 使用例

```{.python.marimo}
_mode1 = Mode(name="Mode1")
_res1 = Resource(name="Res1")
_mode1.addResource(resource=_res1, fixed=1)
_mode1.addCost(fixed=100.0, variable=1.0)
print(_mode1)
```

## Activity


サプライ・チェインとは，資源を時・空間内で消費・生成させることであると捉え，
資源を消費・生成する基本となる単位を活動 (activity) とよぶ．
Activityはサプライ・チェインに必要な諸活動（作業，タスク，ジョブ）を表すクラスである．

活動集合に対しても，点と同様に集約・非集約関係が定義できる．
また，活動は，点もしくは枝上で定義することもできる。
その際には，活動は局所的に定義され，そうでない場合には大域的に定義される。

属性

- atype: 活動（作業）のタイプを表す文字列（'make', 'setup', 'transport', 'inventory', 'shift' など）

      - make: 製造（生産）活動； 付随するモード $m$ に対して，フローを表す変数を定義する。点 $i$ 上で定義されている場合には，変数 $x_{im(t)}^p$ が，枝 $(i,j)$ 上で定義されている場合には，変数 $x_{ijm(t)}^p$ が付加される。ここで$p$ は活動に付随する製品であり，$t$ は多期間モデルの場合の期（リード時間や輸送時間が定義される場合には，その時間を引いた期）である。なお，多期間モデルの場合には，点上に在庫を表す変数が自動的に追加される。なお，枝 $(i,j)$ 上で定義された活動の場合には，付随するモードの部品は点 $i$ で消費され，付随する製品と副生成物は点 $j$ で発生するものとする。
      - setup: 段取りを伴う製造（生産）活動； 付随するモードに対して，フローを表す変数の他に，段取りを表す $0$-$1$ 変数が追加される。
      - transport: 輸送活動；枝 $(i,j)$ 上で定義された活動に対して用いられ，活動に付随する製品が点 $i$ から点 $j$ に移動することを表す。
      - inventory: 在庫活動；モードで定義される部品を用いて，製品を生産し在庫する活動を表す。

- mtype: 活動に付随するモードのタイプ；1つのモードを選択する('select')，比率で配分 ('proportional')，すべてのモードが選択可能 ('all') などがある．
- product: 活動に付随する（生産される，輸送される，在庫される）製品（関連する複数の製品はモードで定義する．）
- modes: 活動を実際に行う方法（モード）を保持する辞書
- nodes: 活動が行われる点の名称の集合
- arcs: 活動が行われる枝の名称の集合

メソッド

- addMode: モードを追加する。
- addModes: モードのリストを一度に追加する。

```{.python.marimo}
# | export
class Activity(Entity):
    atype: Optional[str] = Field(
        description="活動（作業）のタイプ（'make', 'setup', 'inventory', 'transport'など ",
        default="make",
    )
    mtype: Optional[str] = Field(
        description="モードのタイプ（'select', 'proportional', 'all' など ",
        default="select",
    )
    product: Product = Field(description="製品インスタンス", default=None)
    modes: Optional[Dict[str, Mode]] = Field(
        description="モード名をキー，モードインスタンスを値とした辞書",
        default=None,
    )

    nodes: Optional[Set] = Field(
        description="活動を含む点の集合", default=None
    )
    arcs: Optional[Set] = Field(description="活動を含む枝の集合", default=None)

    # OptSeq用
    # duedate: Optional[Union[str,int]]              = Field( description="納期（整数か'Infinity')", default="Infinity" ) #or Period or DateTime
    # backward: Optional[bool]                       = Field( description="後ろ詰めのときTrue", default = False )
    # weight: Optional[int]                          = Field( description="納期遅れペナルティ（重み）", default = 1 )
    # autoselect: Optional[bool]                     = Field( description="モードを自動的に選択するときTrue", default=False )

    def addMode(self, mode: Mode) -> None:
        if self.modes is None:
            self.modes = {}
        self.modes[mode.name] = mode

        if mode.activities is None:
            mode.activities = set([])
        mode.activities.add(self.name)

    def addModes(self, modes: List[Mode]) -> None:
        for m in modes:
            self.addMode(m)
```

### 使用例

```{.python.marimo}
_prod1 = Product(name="Prod1")
_act1 = Activity(name="Act1", product=_prod1)
_act2 = Activity(name="Act2", product=_prod1)
_mode1 = Mode(name="Mode1")
_mode2 = Mode(name="Mode2")
_act1.addMode(_mode1)
_act1.addModes([_mode1, _mode2])
_act2.addMode(_mode1)
print(_act1)
```

## Node

原料供給地点，工場，倉庫の配置可能地点，顧客（群），作業工程，在庫の一時保管場所など，
サプライ・チェインに関わるすべての地点を総称して点 (node) とよぶ．
Nodeは点を表すクラスである．
点集合間には集約・非集約関係が定義できる．たとえば，顧客を集約したものが顧客群となる．


属性
- location: 経度・緯度のタプル
- location_index: Matricesクラスで定義された行列のインデックス
- activities: 点で定義された活動の辞書（名前をキーとし，活動オブジェクトを値とする

メソッド
- addActivity: 活動を追加する。
- addActivities: 活動のリストを一度に追加する。

```{.python.marimo}
# | export
class Node(Entity):
    location: Union[None, Json[Tuple[float, float]], Tuple[float, float]] = (
        Field(description="経度・緯度のタプル", default=None)
    )
    location_index: Union[None, int] = Field(
        description="Matricesクラスで定義された行列のインデックス",
        default=None,
    )
    activities: Optional[Dict[str, Activity]] = Field(
        description="点で定義された活動の辞書（名前をキーとし，活動オブジェクトを値とする）",
        default=None,
    )

    def addActivity(self, activity: Activity) -> None:
        if self.activities is None:
            self.activities = {}
        self.activities[activity.name] = activity
        if activity.nodes is None:
            activity.nodes = set([])
        activity.nodes.add(self.name)

    def addActivities(self, activities: List[Activity]) -> None:
        for a in activities:
            self.addActivity(a)
```

### 使用例

```{.python.marimo}
_node1 = Node(name="Node1", location=(123.45, 567.89))
_node2 = Node(name="Node2")
_prod1 = Product(name="Prod1")
_act1 = Activity(name="Act1", product=_prod1)
_node1.addActivity(_act1)
_node1.model_dump_json(exclude_defaults=True)
```

## Arc

点の対（2つ組）を枝 (arc) とよぶ．
Arcは枝を表すクラスである．枝は，点と点の関係を表し，空間上の移動を表現するために用いられる．

属性

- source: 始点
- sink: 終点
- distance: 距離
- time: 移動時間
- activities: 点で定義された活動の辞書（名前をキーとし，活動オブジェクトを値とする．）

メソッド

- addActivity: 活動を追加する。
- addActivities: 活動のリストを一度に追加する。

```{.python.marimo}
# | export
class Arc(Entity):
    source: Node = Field(description="始点")
    sink: Node = Field(description="始点")
    distance: Optional[Union[int, float]] = Field(
        description="距離", default=None
    )
    time: Optional[Union[int, float]] = Field(description="時間", default=None)

    activities: Optional[Dict[str, Activity]] = Field(
        description="枝で定義された活動の辞書（名前をキーとし，活動オブジェクトを値とする）",
        default=None,
    )

    def addActivity(self, activity: Activity) -> None:
        if self.activities is None:
            self.activities = {}
        self.activities[activity.name] = activity
        if activity.arcs is None:
            activity.arcs = set([])
        activity.arcs.add((self.source.name, self.sink.name))

    def addActivities(self, activities: List[Activity]) -> None:
        for a in activities:
            self.addActivity(a)
```

### 使用例

```{.python.marimo}
_node1 = Node(name="Node1", location=(123.45, 567.89))
_node2 = Node(name="Node2")
_arc1 = Arc(name="arc1", source=_node1, sink=_node2)
_prod1 = Product(name="Prod1")
_act1 = Activity(name="Act1", product=_prod1)
_arc1.addActivity(_act1)
_arc1.model_dump_json(exclude_defaults=True)
```

## Data

Dataは，製品，ノード，期ごとに定義される数値データを保持するクラスである．
名前は必要ないのでEntityクラスから派生させない．
モデル上では，点，製品，期をインデックスとして定義することができる．
インデックスに依存しないデータに対しては $*$（アスタリスク）の文字列を用いて，モデル内に保管される．


属性

- dtype: データのタイプであり，以下から選択する．

      - demand: 需要量
      - supply: 供給量
      - value: 製品の価値
      - inventory: 在庫量

- amount: 量
- std: 標準偏差．ばらつきのあるデータで用いる．（それともscipy.statsのdistributionを指定し，locとscaleパラメータを与えるようにすべきか？）
- over_penalty: 超過ペナルティ；既定値は大きな数で超過を許さない
- under_penalty: 不足ペナルティ；既定値は大きな数で不足を許さない

```{.python.marimo}
# | export
class Data(BaseModel):
    dtype: str = Field(
        description="データのタイプ（demand,supply,value,inventoryから選択)",
        default="demand",
    )
    amount: Optional[Union[int, float]] = Field(
        description="データの値", default=0
    )
    std: Optional[float] = Field(description="標準偏差", default=0.0)
    over_penalty: Optional[Union[int, float]] = Field(
        description="超過ペナルティ", default=999999
    )
    under_penalty: Optional[Union[int, float]] = Field(
        description="不足ペナルティ", default=999999
    )
```

### 使用例

```{.python.marimo}
_data = Data(dtype="demand", amount=100.0, std=10.0)
_data
```

<!-- ## Inventory

Inventoryは在庫関連の情報を保持するクラスである。

＝＞ Inventory情報はModeにもたせ，Activityの方策（atype）を"inventory"に設定する。


属性

- safety_inventory: 安全在庫量（最終期の目標在庫量）
- initial_inventory: 初期在庫量
- target_inventory: 目標在庫量

TODO： 他の在庫モデルのパラメータをここで保持 -->

```{.python.marimo}
# | hide
# class Inventory(Entity):
#     safety_inventory: Optional[float] = Field(description="安全在庫量（最終期の目標在庫量）", default=0.)
#     initial_inventory: Optional[float] = Field(description="初期在庫量", default=0.)
#     target_inventory: Optional[float] = Field(description="目標在庫量", default=0.)
#     #LT, value?
# inv1 = Inventory(name="inv1")
# inv1
```

## Constraint

Constraintは，制約を定義するためのクラスである． スケジューリングモデルに対しては，系全体での資源制約である再生不能資源とも考えられる．

制約は，活動 $a$ がモード $m$ を実行するときに $1$ の $0$-$1$ 変数 $x_{am}$ に対する以下の線形制約として記述される．

$$
 \sum_{a,m} coeff_{am} x_{am} \leq (=, \geq)  rhs
$$

制約インスタンスは，以下のメソッドをもつ．

- setRhs(rhs)は再生不能資源を表す線形制約の右辺定数をrhsに設定する．引数は整数値（負の値も許すことに注意）とする．

- setDirection(dir)は再生不能資源を表す制約の種類をdirに設定する． 引数のdirは'<=', '>=', '='のいずれかとする．

- addTerms(coeffs,vars,values)は，再生不能資源制約の左辺に1つ，もしくは複数の項を追加するメソッドである． 作業がモードで実行されるときに $1$， それ以外のとき $0$ となる変数（値変数）を x[作業,モード]とすると，  追加される項は，
$係数 \times x[作業,モード]$
と記述される． addTermsメソッドの引数は以下の通り．

  - coeffsは追加する項の係数もしくは係数リスト．係数もしくは係数リストの要素は整数（負の値も許す）．
  - varsは追加する項の作業インスタンスもしくは作業インスタンスのリスト． リストの場合には，リストcoeffsと同じ長さをもつ必要がある．
  - valuesは追加する項のモードインスタンスもしくはモードインスタンスのリスト． リストの場合には，リストcoeffsと同じ長さをもつ必要がある．


制約インスタンスは以下の属性をもつ．

- nameは制約名である．
- rhsは制約の右辺定数である． 既定値は $0$．
- directionは制約の方向を表す．　既定値は '<='．
- termsは制約の左辺を表す項のリストである．各項は (係数,活動インスタンス,モードインスタンス) のタプルである．
- weightは制約を逸脱したときのペナルティの重みを表す． 正数値か絶対制約を表す'inf'を入れる． 既定値は無限大（絶対制約）を表す文字列'inf'である．

```{.python.marimo}
# | export
class Constraint(Entity):
    rhs: Optional[Union[int, float]] = Field(description="右辺定数", default=0)
    direction: Optional[str] = Field(description="制約の方向", default="<=")
    # terms: Optional[List]                     = Field( description="制約の左辺項のリスト", default = None )
    terms: Optional[List[Tuple[Union[int, float], Activity, Mode]]] = Field(
        description="制約の左辺項のリスト", default=None
    )
    weight: Optional[Union[int, float, str]] = Field(
        description="制約の逸脱ペナルティ", default="inf"
    )

    def addTerms(self, coeffs=None, vars=None, values=None):
        """
        Add new terms into left-hand-side of nonrenewable resource constraint.

            - Arguments:
                - coeffs: Coefficients for new terms; either a list of coefficients or a single coefficient.
                The three arguments must have the same size.
                - vars: Activity objects for new terms; either a list of activity objects or a single activity object.
                The three arguments must have the same size.
                - values: Mode objects for new terms; either a list of mode objects or a single mode object.
                The three arguments must have the same size.

            - Example usage:

            >>> budget.addTerms(1,act,express)

            adds one unit of nonrenewable resource (budget) if activity "act" is executed in mode "express."

        """
        if self.terms is None:
            self.terms = []
        if type(coeffs) != type([]):
            self.terms.append((coeffs, vars, values))
        elif (
            type(coeffs) != type([])
            or type(vars) != type([])
            or type(values) != type([])
        ):
            print("coeffs, vars, values must be lists")
            raise TypeError("coeffs, vars, values must be lists")
        elif (
            len(coeffs) != len(vars)
            or len(coeffs) != len(values)
            or len(values) != len(vars)
        ):
            print("length of coeffs, vars, values must be identical")
            raise TypeError("length of coeffs, vars, values must be identical")
        else:
            for i in range(len(coeffs)):
                self.terms.append((coeffs[i], vars[i], values[i]))

    def setRhs(self, rhs=0):
        """
        Sets the right-hand-side of linear constraint.

            - Argument:
                - rhs: Right-hand-side of linear constraint.

            - Example usage:

            >>> L.setRhs(10)

        """
        self.rhs = rhs

    def setDirection(self, direction="<="):
        if direction in ["<=", ">=", "="]:
            self.direction = direction
        else:
            print(
                "direction setting error; direction should be one of '<=' or '>=' or '='"
            )
            raise NameError(
                "direction setting error; direction should be one of '<=' or '>=' or '='"
            )

    def printConstraint(self):
        """
        Returns the information of the linear constraint.

        The constraint is expanded and is shown in a readable format.
        """

        f = [f"Constraint {self.name}: weight={self.weight}: "]
        if self.direction == ">=" or self.direction == ">":
            for coeff, var, value in self.terms:
                f.append("{0}({1},{2}) ".format(-coeff, var.name, value.name))
            f.append("<={0} \n".format(-self.rhs))
        elif self.direction == "==" or self.direction == "=":
            for coeff, var, value in self.terms:
                f.append("{0}({1},{2}) ".format(coeff, var.name, value.name))
            f.append("<={0} \n".format(self.rhs))
            f.append("nonrenewable weight {0} ".format(self.weight))
            for coeff, var, value in self.terms:
                f.append("{0}({1},{2}) ".format(-coeff, var.name, value.name))
            f.append("<={0} \n".format(-self.rhs))
        else:
            for coeff, var, value in self.terms:
                f.append("{0}({1},{2}) ".format(coeff, var.name, value.name))
            f.append("<={0} \n".format(self.rhs))

        return "".join(f)
```

### 使用例

```{.python.marimo}
_con1 = Constraint(name="constraint1")
_mode1 = Mode(name="Mode1")
_act1 = Activity(name="Act1")
_act1.addMode(_mode1)
_con1.addTerms(coeffs=1, vars=_act1, values=_mode1)
_con1.printConstraint()
```

## Model

Modelは上の諸Entityを組み合わせたモデルクラスである。データ入力後にupdateメソッドで最適化の準備を行う．

属性

- activities: 活動情報を保持する辞書（名前をキーとし，活動オブジェクトを値とする；以下同様）
- modes: モード情報を保持する辞書
- resources: 資源情報を保持する辞書
- products: 製品情報を保持する辞書
- periods: 期情報を保持する辞書
- nodes: 点情報を保持する辞書
- arcs: 枝情報を保持する辞書
- constraints: 制約情報を保持する辞書
- data: データ（需要，価値，在庫など）を保持する辞書
- interest_rate: 投資利益率（在庫費用の計算で用いる）

また，上の諸情報をモデルに付加するメソッドをもつ。

最適化メソッドの引数で，最適化モデルの種類を渡す（'lnd', 'optseq', 'risk', 'ssa', 'shift', ... )

```{.python.marimo}
# | export
class Model(Entity):
    activities: Optional[Dict[str, Activity]] = Field(
        description="活動の辞書（名前をキーとし，活動オブジェクトを値とする）",
        default=None,
    )
    modes: Optional[Dict[str, Mode]] = Field(
        description="モードの辞書（名前をキーとし，モードオブジェクトを値とする）",
        default=None,
    )
    resources: Optional[Dict[str, Resource]] = Field(
        description="資源の辞書（名前をキーとし，資源オブジェクトを値とする）",
        default=None,
    )
    products: Optional[Dict[str, Product]] = Field(
        description="製品の辞書（名前をキーとし，製品オブジェクトを値とする）",
        default=None,
    )
    periods: Optional[Dict[str, Period]] = Field(
        description="期の辞書（名前をキーとし，期オブジェクトを値とする）",
        default=None,
    )
    nodes: Optional[Dict[str, Node]] = Field(
        description="点の辞書（名前をキーとし，点オブジェクトを値とする）",
        default=None,
    )
    arcs: Optional[Dict[Tuple[str,str], Arc]] = Field(
        description="枝の辞書（始点名と終点名のタプルをキーとし，枝オブジェクトを値とする）",
        default=None,
    )
    constraints: Optional[Dict[str, Constraint]] = Field(
        description="制約の辞書（名前をキーとし，制約オブジェクトを値とする）",
        default=None,
    )

    data: Optional[Dict] = Field(
        description="数値データを保管する辞書（データタイプ，製品名，点名，期名をキーとし，データインスタンスを値とする）",
        default=None,
    )
    # demands: Optional[Dict]                      = Field( description="需要の辞書（製品名，点名，期名をキーとし，需要量を値とする）", default=None )
    # values: Optional[Dict]                       = Field( description="価値の辞書（製品名，点名，期名をキーとし，価値を値とする）", default=None )
    # inventories: Optional[Dict]                  = Field( description="在庫の辞書（製品名，点名，期名をキーとし，在庫オブジェクトを値とする））", default=None )
    interest_rate: Optional[float] = Field(
        description="投資利子率（在庫費用の計算で用いる）", default=None
    )

    period_list: Optional[List[Tuple]] = Field(
        description="期のリスト（開始時刻と期の名前のタプルを要素とする）",
        default=None,
    )
    period_index: Optional[Dict[str, int]] = Field(
        description="期の名前をキー，期インデックスを値とした辞書",
        default=None,
    )

    def addResource(
        self,
        name: str,
        capacity: Optional[Union[int, Dict]] = None,
        fixed_cost: Optional[Union[int, float]] = 0,
    ) -> Resource:
        if self.resources is None:
            self.resources = {}
        self.resources[name] = Resource(
            name=name, capacity=capacity, fixed_cost=fixed_cost
        )
        return self.resources[name]

    def addActivity(
        self, name: str, atype: Optional[str] = "make", product: Product = None
    ) -> Activity:
        if self.activities is None:
            self.activities = {}
        self.activities[name] = Activity(
            name=name, atype=atype, product=product
        )
        # 関連するモードをモデルに追加する？ addModeで追加すると二度手間になる！
        return self.activities[name]

    # 必ず活動に付随させるので不必要？
    def addMode(
        self,
        name: str,
        components: Optional[Dict[str, float]] = None,
        byproducts: Optional[Dict[str, float]] = None,
        fixed_requirement: Optional[Dict[str, float]] = None,
        variable_requirement: Optional[Dict[str, float]] = None,
        fixed_cost: Optional[float] = None,
        variable_cost: Optional[float] = None,
    ) -> Mode:
        if self.modes is None:
            self.modes = {}
        self.modes[name] = Mode(
            name=name,
            components=components,
            byproduct=byproducts,
            fixed_requirement=fixed_requirement,
            variable_requirement=variable_requirement,
            fixed_cost=fixed_cost,
            variable_cost=variable_cost,
        )
        return self.modes[name]

    def addNode(
        self,
        name: str,
        location: Union[
            None, Json[Tuple[float, float]], Tuple[float, float]
        ] = None,
    ) -> Node:
        if self.nodes is None:
            self.nodes = {}
        self.nodes[name] = Node(name=name, location=location)
        return self.nodes[name]

    def addArc(self, name: str, source: Node, sink: Node) -> Arc:
        if self.arcs is None:
            self.arcs = {}
        if self.nodes is None:
            self.nodes = {}
        self.arcs[source.name, sink.name] = Arc(
            name=name, source=source, sink=sink
        )
        if source.name not in self.nodes:
            self.nodes[source.name] = source
        if sink.name not in self.nodes:
            self.nodes[sink.name] = sink
        return self.arcs[source.name, sink.name]

    def addPeriod(
        self,
        name: str,
        start: Optional[Union[int, float, datetime, date, time]] = None,
    ):
        if self.periods is None:
            self.periods = {}
        self.periods[name] = Period(name=name, start=start)
        return self.periods[name]

    def addProduct(
        self,
        name: str,
        volume: float = None,
        weight: float = None,
        value: float = None,
    ):
        if self.products is None:
            self.products = {}
        self.products[name] = Product(
            name=name, volume=volume, weight=weight, value=value
        )
        return self.products[name]

    def addConstraint(
        self,
        name: str,
        rhs: Union[int, float] = 0,
        direction: str = "<=",
        weight: Union[int, float, str] = "inf",
    ):
        if self.constraints is None:
            self.constraints = {}
        self.constraints[name] = Constraint(
            name=name, rhs=rhs, direction=direction, weight=weight
        )
        return self.constraints[name]

    def addData(
        self,
        dtype: str,
        product: Product,
        node: Optional[Node] = None,
        period: Optional[Period] = None,
        amount: Union[int, float] = 0,
        std: Union[int, float] = 0,
        under_penalty: Union[int, float] = 999999,
        over_penalty: Union[int, float] = 999999,
    ) -> None:
        if self.data is None:
            self.data = {}
        if node is None:
            node_name = "*"
        else:
            node_name = node.name
        if period is None:
            period_name = "*"
        else:
            period_name = period.name
        self.data[dtype, product.name, node_name, period_name] = Data(
            dtype=dtype,
            amount=amount,
            std=std,
            under_penalty=under_penalty,
            over_penalty=over_penalty,
        )

    def update(self):
        if self.periods is not None:
            self.period_list = [
                (t.start, name) for (name, t) in self.periods.items()
            ]
            self.period_list.sort()
            self.period_index = {}
            for i, (t, name) in enumerate(self.period_list):
                self.period_index[name] = i
        else:
            self.period_list = [(datetime.now(), "period(0)")]
        # return self.period_list

    # def addDemand(self, product:Product, node:Optional[Node]=None, period:Optional[Period]=None, amount:Union[int,float]=0,
    #                     std:Union[int,float]=0, under_penalty:Union[int,float]=999999, over_penalty:Union[int,float]=999999)->None:
    #     if self.demands is None:
    #         self.demands = {}
    #     if node is None:
    #         node_name = "*"
    #     else:
    #         node_name = node.name
    #     if period is None:
    #         period_name = "*"
    #     else:
    #         period_name =period.name
    #     self.demands[product.name, node_name, period_name] = amount

    # def addValue(self, product:Product, node:Optional[Node]=None, period:Optional[Period]=None, value:Union[int,float]=0)->None:
    #     if self.values is None:
    #         self.values = {}
    #     if node is None:
    #         node_name = "*"
    #     else:
    #         node_name = node.name
    #     if period is None:
    #         period_name = "*"
    #     else:
    #         period_name =period.name
    #     self.values[product.name, node_name, period_name] = value

    # def addInventory(self, product:Product, node:Optional[Node]=None, period:Optional[Period]=None,
    #                  amount:Union[int,float] = 0
    #                 )->None:
    #     if self.inventories is None:
    #         self.inventories = {}
    #     if node is None:
    #         node_name = "*"
    #     else:
    #         node_name = node.name
    #     if period is None:
    #         period_name = "*"
    #     else:
    #         period_name =period.name
    #     self.inventories[product.name, node_name, period_name] = amount

    def __str__(self):
        """
        SCMLフォーマットを定義して出力
        """
        return ""
```

### クラウドソリューション

```{mermaid}
sequenceDiagram
  participant Client
  participant Cloud
  loop
    Cloud->>Cloud: Optimization
  end
  Note right of Cloud: Scheduling <br/> Lot-sizing <br/> Logistics Network Design <br/> Shift <br/> Vehicle Routing
  Client->>Cloud: Model Instance in JSON
  Cloud->>Client: Solution in JSON
```
<!---->
### 使用例

```{.python.marimo}
_model1 = Model(name="model1")

_node1 = Node(name="Node1", location=(123.45, 567.89))
_period1 = Period(name="Period1", start="2023-04-23T10:20:30.400+02:30")
_prod1 = Product(name="Prod1")

_model1.addData(dtype="demand", product=_prod1, amount=100.0)
_model1.addData(
    dtype="supply", product=_prod1, node=_node1, period=_period1, amount=100.0
)
_model1.addNode(name="Node1")
_con1 = _model1.addConstraint(name="constraint1")
_con1.setRhs(100)
_model1.model_dump_json(exclude_none=True)
```

```{.python.marimo}
_model2 = Model(name="model2")
for _i, _t in enumerate(
    pd.date_range(start="2024/1/1", end="2024/1/31", freq="w")
):
    _model2.addPeriod(name=f"period({_i})", start=_t)
_model2.update()
print(_model2.period_index)
_model2.model_dump_json(exclude_none=True)
```

## Optimize関数

フローの最適化を行う関数

TODO: 結果をクラスの属性に記入する

モデルの種類によって呼ぶ関数を分岐

```{.python.marimo}
gp
```

```{.python.marimo}
def optimize(
    model: Model, start_period_name: str = None, end_period_name: str = None
) -> gp.Model:
    if (
        model.periods is None
        or start_period_name is None
        or end_period_name is None
    ):
        start = 0
        end = 1
    else:
        start = model.period_index[start_period_name]
        end = model.period_index[end_period_name] + 1
    assert start < end
    print("計画期間\u3000=", start, end)
    gp_model = gp.Model()
    x, y, Y = ({}, {}, {})
    I = {}
    vc, fc = ({}, {})
    rfc = {}
    ub = {}
    demand_point, supply_point = ({}, {})
    value, inventory = ({}, {})
    over_penalty, under_penalty = ({}, {})
    total_demand = defaultdict(float)
    total_supply = defaultdict(float)
    for dtype, p, i, t in model.data:
        if t == "*":
            for t0 in range(start, end):
                if dtype == "demand":
                    demand_point[p, i, t0] = model.data[dtype, p, i, t].amount
                    total_demand[p] = total_demand[p] + demand_point[p, i, t0]
                elif dtype == "supply":
                    supply_point[p, i, t0] = model.data[dtype, p, i, t].amount
                    total_supply[p] = total_supply[p] + supply_point[p, i, t0]
                elif dtype == "value":
                    value[p, i, t0] = model.data[dtype, p, i, t].amount
                elif dtype == "inventory":
                    inventory[p, i, t0] = model.data[dtype, p, i, t].amount
                over_penalty[p, i, t0] = model.data[
                    dtype, p, i, t
                ].over_penalty
                under_penalty[p, i, t0] = model.data[
                    dtype, p, i, t
                ].under_penalty
        else:
            period_idx = model.period_index[t]
            if dtype == "demand":
                demand_point[p, i, t] = model.data[dtype, p, i, t].amount
                total_demand[p] = total_demand[p] + demand_point[p, i, t]
            elif dtype == "supply":
                supply_point[p, i, t] = model.data[dtype, p, i, t].amount
                total_supply[p] = total_supply[p] + supply_point[p, i, t]
            elif dtype == "value":
                value[p, i, t] = model.data[dtype, p, i, t].amount
            elif dtype == "inventory":
                inventory[p, i, t] = model.data[dtype, p, i, t].amount
            if period_idx >= start and period_idx < end:
                over_penalty[p, i, t] = model.data[dtype, p, i, t].over_penalty
                under_penalty[p, i, t] = model.data[
                    dtype, p, i, t
                ].under_penalty
    lhs = defaultdict(list)
    rhs = defaultdict(list)
    fixed_req = defaultdict(list)
    variable_req = defaultdict(list)
    for i, node in model.nodes.items():
        if node.activities is not None and len(node.activities) > 0:
            for aname, act in node.activities.items():
                if act.atype == "make":
                    p = act.product.name
                    for t in range(start, end):
                        for m, mode in act.modes.items():
                            if mode.upper_bound is None:
                                ub[i, m, p] = max(
                                    total_demand[p], total_supply[p]
                                )
                            else:
                                ub[i, m, p] = float(mode.upper_bound)
                            x[i, m, p, t] = gp_model.addVar(
                                name=f"x({i},{m},{p},{t})", vtype="C"
                            )
                            y[i, m, p, t] = gp_model.addVar(
                                name=f"y({i},{m},{p},{t})", vtype="B"
                            )
                            if mode.components is not None:
                                for q, quantity in mode.components.items():
                                    rhs[i, q, t].append(
                                        (quantity, x[i, m, p, t])
                                    )
                            if mode.byproducts is not None:
                                for q, quantity in mode.byproducts.items():
                                    lhs[j, q, t].append(
                                        (quantity, x[i, m, p, t])
                                    )
                            lhs[i, p, t].append((1.0, x[i, m, p, t]))
                            vc[i, m, p] = mode.variable_cost
                            fc[i, m, p] = mode.fixed_cost
                            if mode.fixed_requirement is not None:
                                for r, val in mode.fixed_requirement.items():
                                    fixed_req[r].append((val, y[i, m, p, t]))
                            if mode.variable_requirement is not None:
                                for (
                                    r,
                                    val,
                                ) in mode.variable_requirement.items():
                                    variable_req[r].append(
                                        (val, x[i, m, p, t])
                                    )
    for (i, j), arc in model.arcs.items():
        if arc.activities is not None and len(arc.activities) > 0:
            for a, act in arc.activities.items():
                p = act.product.name
                for m, mode in act.modes.items():
                    if mode.upper_bound is None:
                        ub[i, j, m, p] = max(total_demand[p], total_supply[p])
                    else:
                        ub[i, j, m, p] = float(mode.upper_bound)
                    fc[i, j, m, p] = mode.fixed_cost
                    vc[i, j, m, p] = mode.variable_cost
                    for t in range(start, end):
                        x[i, j, m, p, t] = gp_model.addVar(
                            name=f"x({i},{j},{m},{p},{t})"
                        )
                        y[i, j, m, p, t] = gp_model.addVar(
                            name=f"y({i},{j},{m},{p},{t})", vtype="B"
                        )
                        if act.atype == "transport":
                            rhs[i, p, t].append((1.0, x[i, j, m, p, t]))
                            lhs[j, p, t].append((1.0, x[i, j, m, p, t]))
                        elif act.atype == "make":
                            if mode.components is not None:
                                for q, quantity in mode.components.items():
                                    rhs[i, q, t].append(
                                        (quantity, x[i, j, m, p, t])
                                    )
                            if mode.byproducts is not None:
                                for q, quantity in mode.byproducts.items():
                                    lhs[j, q, t].append(
                                        (quantity, x[i, j, m, p, t])
                                    )
                            lhs[j, p, t].append((1.0, x[i, j, m, p, t]))
                        fc[i, j, m, p, t] = mode.fixed_cost
                    if mode.fixed_requirement is not None:
                        for r, val in mode.fixed_requirement.items():
                            fixed_req[r].append((val, y[i, j, m, p, t]))
                    if mode.variable_requirement is not None:
                        for r, val in mode.variable_requirement.items():
                            variable_req[r].append((val, x[i, j, m, p, t]))
    for i, node in model.nodes.items():
        if node.activities is None:
            continue
        for a, act in node.activities.items():
            if act.atype == "inventory":
                p = act.product.name
                for t in range(start, end):
                    I[i, p, t] = gp_model.addVar(name=f"I({i},{p},{t})")
    for (p, i, t), amount in inventory.items():
        I[i, p, model.period_index[t]] = amount
    slack, surplus = ({}, {})
    for p, i, t in demand_point:
        slack[i, p, t] = gp_model.addVar(name=f"slack({i},{p},{t})")
        surplus[i, p, t] = gp_model.addVar(name=f"surplus({i},{p},{t})")
    for p, i, t in supply_point:
        slack[i, p, t] = gp_model.addVar(name=f"slack({i},{p},{t})")
        surplus[i, p, t] = gp_model.addVar(name=f"surplus({i},{p},{t})")
    if model.resources is not None:
        for r, res in model.resources.items():
            for t in range(start, end):
                rfc[r, t] = res.fixed_cost
                Y[r, t] = gp_model.addVar(name=f"Y({r},{t})", vtype="B")
    if GUROBI:
        gp_model.update()
    for t in range(start, end):
        for i, node in model.nodes.items():
            for p, product in model.products.items():
                if len(lhs[i, p, t]) == 0 and len(rhs[i, p, t]) == 0:
                    continue
                if (p, i, t) in demand_point:
                    gp_model.addConstr(
                        gp.quicksum(
                            (
                                coeff * variable
                                for coeff, variable in lhs[i, p, t]
                            )
                        )
                        - gp.quicksum(
                            (
                                coeff * variable
                                for coeff, variable in rhs[i, p, t]
                            )
                        )
                        - slack[i, p, t]
                        + surplus[i, p, t]
                        == demand_point[p, i, t],
                        name=f"flow_cons({i},{p},{t})",
                    )
                elif (p, i, t) in supply_point:
                    gp_model.addConstr(
                        -gp.quicksum(
                            (
                                coeff * variable
                                for coeff, variable in lhs[i, p, t]
                            )
                        )
                        + gp.quicksum(
                            (
                                coeff * variable
                                for coeff, variable in rhs[i, p, t]
                            )
                        )
                        + slack[i, p, t]
                        - surplus[i, p, t]
                        == supply_point[p, i, t],
                        name=f"flow_cons({i},{p},{t})",
                    )
                else:
                    gp_model.addConstr(
                        gp.quicksum(
                            (
                                coeff * variable
                                for coeff, variable in lhs[i, p, t]
                            )
                        )
                        - gp.quicksum(
                            (
                                coeff * variable
                                for coeff, variable in rhs[i, p, t]
                            )
                        )
                        + (I[i, p, t - 1] if (i, p, t - 1) in I else 0.0)
                        - (I[i, p, t] if (i, p, t) in I else 0.0)
                        == 0,
                        name=f"flow_cons({i},{p},{t})",
                    )
        if model.resources is not None:
            for r, res in model.resources.items():
                gp_model.addConstr(
                    gp.quicksum(
                        (val * variable for val, variable in variable_req[r])
                    )
                    + gp.quicksum(
                        (val * variable for val, variable in fixed_req[r])
                    )
                    <= res.capacity * Y[r, t],
                    name=f"capacity({r},{t})",
                )
    for idx in y:
        gp_model.addConstr(
            x[idx] <= ub[idx[:-1]] * y[idx], name=f"connection({idx})"
        )
    gp_model.setObjective(
        gp.quicksum((vc[idx[:-1]] * x[idx] for idx in x))
        + gp.quicksum((fc[idx[:-1]] * y[idx] for idx in y))
        + gp.quicksum((rfc[idx] * Y[idx] for idx in Y))
        + gp.quicksum(
            (
                under_penalty[p, i, t] * slack[i, p, t]
                + over_penalty[p, i, t] * surplus[i, p, t]
                for i, p, t in surplus
            )
        )
        + gp.quicksum(
            (
                model.interest_rate * value[p, i, t] * I[i, p, t]
                for i, p, t in I
                if t >= start and t < end
            )
        ),
        gp.GRB.MINIMIZE,
    )
    gp_model.optimize()
    print("Opt. Val=", gp_model.ObjVal)
    return gp_model
```

## (Network) Visualize関数

モデルの可視化を行う関数

```{.python.marimo}
class SCMGraph(nx.DiGraph):
    """
    SCMGraph is a class of directed graph with edge weight that can be any object.
    I just use the functions such as in_degree, out_degree, successors,
    predecessors, in_edges_iter, out_edges_iter.
    So it is convertible to any graph class that can access the adjacent nodes more quickly.
    """

    def random_directed_tree(self, n=1, seed=None):
        """
        generate random directed tree
        """
        random.seed(seed)
        G = nx.generators.trees.random_tree(n=n, seed=seed)
        self.add_nodes_from(G)
        for u, v in G.edges():
            if random.random() <= 0.5:
                self.add_edge(u, v)
            else:
                self.add_edge(v, u)

    def layered_network(self, num_in_layer=None, p=0.5, seed=None):
        """
        Input the number of nodes in layers as a list like NumInLayer=[4,5,6,7]
        and the probability with which edge occures,
        return a layered and connected directed graph
        """
        random.seed(seed)
        if num_in_layer is None:
            num_in_layer = [1, 1]
        else:
            num_in_layer = list(num_in_layer)
        Layer = []
        startID = 0
        layerID = {}
        numlayer = 0
        for l in num_in_layer:
            endID = startID + l
            Layer.append(range(startID, endID))
            for i in range(startID, endID):
                layerID[i] = numlayer
            numlayer = numlayer + 1
            startID = endID
        n = endID
        self.add_nodes_from(range(n))
        for l in range(len(Layer) - 1):
            for i in Layer[l]:
                for j in Layer[l + 1]:
                    if random.random() <= p:
                        self.add_edge(i, j)

    def layout(self):
        """
        Compute x,y coordinates for the supply chain
           The algorithm is based on a simplified version of Sugiyama's method.
           First assign each node to the (minimum number of) layers;
           Then compute the y coordinate by computing the means of y-values
           of adjacent nodes
           return the dictionary of (x,y) positions of the nodes
        """
        longest_path = nx.dag_longest_path(self)
        LayerLB = {}
        pos = {}
        MaxLayer = len(longest_path)
        candidate = set([i for i in self]) - set(longest_path)
        for i in candidate:
            LayerLB[i] = 0
        Layer = defaultdict(list)
        for i, v in enumerate(longest_path):
            Layer[i] = [v]
            LayerLB[v] = i
            for w in self.successors(v):
                if w in candidate:
                    LayerLB[w] = LayerLB[v] + 1
        L = list(nx.topological_sort(self))
        for v in L:
            if v in candidate:
                Layer[LayerLB[v]].append(v)
                candidate.remove(v)
                for w in self.successors(v):
                    if w in candidate:
                        LayerLB[w] = max(LayerLB[v] + 1, LayerLB[w])
        MaxLayer = len(Layer)
        for i in range(MaxLayer + 1):
            if i == 0:
                j = 0
                for v in Layer[i]:
                    pos[v] = (i, j)
                    j = j + 1
            else:
                tmplist = []
                for v in Layer[i]:
                    sumy = 0.0
                    j = 0.0
                    for w in self.predecessors(v):
                        ii, jj = pos[w]
                        sumy = sumy + jj
                        j = j + 1.0
                    if j != 0:
                        temp = sumy / j
                    else:
                        temp = j
                    tmplist.append((temp, v))
                tmplist.sort()
                order = [v for _, v in tmplist]
                j = 0
                for v in Layer[i]:
                    pos[order[j]] = (i, j)
                    j = j + 1
        return pos

    def down_order(self):
        """
        generator fuction in topological order
        generate the order of nodes from suppliers to demand points
        """
        degree0 = []
        degree = {}
        for v in self:
            if self.in_degree(v) == 0:
                degree0.append(v)
            else:
                degree[v] = self.in_degree(v)
        while degree0:
            v = degree0.pop()
            yield v
            for w in self.successors(v):
                degree[w] = degree[w] - 1
                if degree[w] == 0:
                    degree0.append(w)

    def up_order(self):
        """
        Generator fuction in the reverse topological order
        generate the order of nodes from to demand points to suppliers
        """
        degree0 = []
        degree = {}
        for v in self:
            if self.out_degree(v) == 0:
                degree0.append(v)
            else:
                degree[v] = self.out_degree(v)
        while degree0:
            v = degree0.pop()
            yield v
            for w in self.predecessors(v):
                degree[w] = degree[w] - 1
                if degree[w] == 0:
                    degree0.append(w)

    def dp_order(self):
        """
        Generater function for the safety stock allocation problem
        This function returns (yields) the leaf ordering sequence of nodes
        Remark: the graph must be tree! Otherwise, this function does not generate
        all the nodes.
        """
        is_tree = nx.is_tree(self)
        if is_tree == False:
            print("Graph is not a tree.")
            return
        Leaf = set([])
        Searched = set([])
        degree = {}
        for v in self:
            degree[v] = self.out_degree(v) + self.in_degree(v)
            if degree[v] <= 1:
                Leaf.add(v)
        while Leaf:
            v = Leaf.pop()
            yield v
            Searched.add(v)
            for w in set(self.successors(v)) | set(self.predecessors(v)):
                if w not in Searched:
                    if degree[w] >= 2:
                        degree[w] = degree[w] - 1
                    if degree[w] <= 1:
                        Leaf.add(w)

    def bfs(self, start=None):
        """
        breadth first search from a given node 'start'
        """
        if start is None:
            start = list(self.nodes)[0]
        L = []
        L.append(start)
        Searched = set([start])
        while L:
            v = L.pop(0)
            yield v
            for w in self.successors(v):
                if w not in Searched:
                    L.append(w)
                    Searched.add(w)
                else:
                    print(f"arc ({v}, {w}) makes a cycle")

    def find_ancestors(self):
        """
        find the ancestors based on the BOM graph
        The set of ancestors of node i is the set of nodes that are reachable from node i (including i).
        """
        ancestors = {v: set([]) for v in self}
        for v in self.up_order():
            ancestors[v] = ancestors[v] | set([v])
            for w in self.successors(v):
                ancestors[v] = ancestors[v] | ancestors[w]
        return ancestors
```

```{.python.marimo}
# | export
def visualize_network(model: Model):
    D = SCMGraph()
    for (i, j), arc in model.arcs.items():
        D.add_edge(i, j)

    pos = {}
    no_position = False
    for i, node in model.nodes.items():
        if node.location is None:
            no_position = True
            break
        else:
            pos[i] = node.location
    if no_position:
        if nx.is_directed_acyclic_graph(D):
            pos = D.layout()
        else:
            pos = nx.spring_layout(D)

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(D, pos=pos, with_labels=True, node_color="Yellow")
    plt.gca()
```

```{.python.marimo}
_node1 = Node(name="Node1", location=(123.45, 567.89))
_node2 = Node(name="Node2")
_model1 = Model(name="model1")
_model1.addArc(name="arc1", source=_node1, sink=_node2)
visualize_network(_model1)
plt.gca()
```

## Graphvizで可視化

```{.python.marimo}
def visualize(model: Model, rankdir: str = "LR", size: float = 5):
    def draw_activity(
        a: str,
        act: Activity,
        g: graphviz.Digraph,
        c: graphviz.Digraph.subgraph,
    ):
        c.attr(style="filled", color="lightgrey")
        c.node(name=a, shape="rectangle", color="red")
        c.node(name=f"{a}\n {act.product.name}", shape="oval", color="yellow")
        if act.modes is not None:
            for m, mode in act.modes.items():
                c.node(name=mode.name, shape="box3d", color="blue")
                c.edge(a, mode.name, arrowhead="tee", style="dotted", color="blue")
                if mode.variable_requirement is not None:
                    requirement = mode.variable_requirement
                elif mode.fixed_requirement is not None:
                    requirement = mode.fixed_requirement
                else:
                    requirement = {}
                for rname, val in requirement.items():
                    g.edge(
                        mode.name,
                        rname,
                        arrowhead="box",
                        style="dashed",
                        color="green",
                    )

    g = graphviz.Digraph("G", filename="ｓｃｍｌ.gv")
    g.graph_attr["rankdir"] = rankdir
    g.graph_attr["size"] = str(size)
    act_in_node = set([])
    if model.nodes is not None:
        for i, node in model.nodes.items():
            with g.subgraph(name=f"cluster[{i}]") as nc:
                nc.attr(style="filled", color="lightblue")
                nc.node(name=str(i), shape="folder", color="black")
                if node.activities is not None:
                    for a, act in node.activities.items():
                        act_in_node.add(a)
                        with nc.subgraph(name=f"cluster[{a}]") as c:
                            draw_activity(a, act, g, c)
    if model.arcs is not None:
        for (i, j), arc in model.arcs.items():
            label = ""
            if arc.activities is not None:
                for a, act in arc.activities.items():
                    label = label + f"{a} \n"
            g.edge(str(i), str(j), arrowhead="normal", label=label)
    if model.resources is not None:
        for r, res in model.resources.items():
            g.node(name=r, shape="trapezium", color="green")
    if model.activities is not None:
        for a, act in model.activities.items():
            if a in act_in_node:
                continue
            with g.subgraph(name=f"cluster[{a}]") as c:
                draw_activity(a, act, g, c)
    bom = graphviz.Digraph("BOM", filename="bom.gv")
    bom.graph_attr["rankdir"] = "LR"
    if model.activities is not None:
        for a, act in model.activities.items():
            bom.node(name=f"{act.product.name}", shape="oval", color="black")
            if act.modes is not None:
                for m, mode in act.modes.items():
                    if mode.components is not None:
                        for child, weight in mode.components.items():
                            if weight != 1.0:
                                label = str(weight)
                            else:
                                label = ""
                            bom.edge(
                                child,
                                act.product.name,
                                arrowhead="curve",
                                label=label,
                            )
    return (g, bom)
```

```{.python.marimo}
# | hide
# def draw_activity(a:str, act:Activity, g:graphviz.Digraph, c:graphviz.Digraph.subgraph ):
#     c.attr(style='filled', color='lightgrey')
#     c.node(name= a, shape="rectangle", color="red")
#     #product
#     c.node(name = f"{a}\n {act.product.name}", shape ="oval", color ="yellow")
#     for m, mode in act.modes.items():
#         c.node(name=mode.name, shape="box3d", color="blue")
#         c.edge(a, mode.name, arrowhead="tee", style="dotted", color="blue" )
#         if mode.variable_requirement is not None:
#             requirement = mode.variable_requirement
#         elif mode.fixed_requirement is not None:
#             requirement = mode.fixed_requirement
#         else:
#             requirement = {}
#         for rname, val in requirement.items():
#             g.edge(mode.name, rname, arrowhead="box", style="dashed", color="green")

# g = graphviz.Digraph('G', filename='ｓｃｍｌ.gv')
# g.graph_attr["rankdir"]= "LR"
# g.graph_attr["size"] = str(50)
# act_in_node = set([])
# if model.nodes is not None:
#     for i, node in model.nodes.items():
#         with g.subgraph(name=f"cluster[{i}]") as nc:
#             nc.attr(style='filled', color='lightblue')
#             nc.node(name= str(i), shape="folder", color="black")
#             if node.activities is not None:
#                 for a, act in node.activities.items():
#                     act_in_node.add(a)
#                     with nc.subgraph(name=f"cluster[{a}]") as c:
#                         draw_activity(a,act,g,c)

# if model.arcs is not None:
#     for (i,j), arc in  model.arcs.items():
#         label = ""
#         if arc.activities is not None:
#             for a, act in arc.activities.items():
#                 label += f"{a} \n"
#         g.edge(str(i), str(j), arrowhead="normal", label=label )

# if model.resources is not None:
#     for r, res in model.resources.items():
#         g.node(name=r, shape="trapezium", color="green")

# if model.activities is not None:
#     for a, act in model.activities.items():
#         if a in act_in_node:
#             continue
#         with g.subgraph(name=f"cluster[{a}]") as c:
#             draw_activity(a,act,g,c)
# #BOM
# bom = graphviz.Digraph('BOM', filename='bom.gv')
# bom.graph_attr["rankdir"]="LR"
# if model.activities is not None:
#     for a, act in model.activities.items():
#         bom.node(name = f"{act.product.name}", shape ="oval", color ="black")
#         for m, mode in act.modes.items():
#             #bom.edge(a, mode.name, arrowhead="tee", style="dotted", color="blue" )
#             if mode.components is not None:
#                 for child, weight  in mode.components.items():
#                     if weight !=1.0:
#                         label = str(weight)
#                     else:
#                         label = ""
#                     bom.edge(child, act.product.name, arrowhead="curve", label= label )
```

## Example: Transportation

いま，顧客数を $n$，工場数を $m$ とし，
顧客を $i=1,2,\ldots,n$，工場を $j=1,2,\ldots,m$ と番号で表すものとする．
また，顧客の集合を $I=\{1,2,\ldots,n \}$，工場の集合を $J=\{ 1,2,\ldots,m \}$ とする．
顧客 $i$ の需要量を $d_i$，顧客 $i$ と施設 $j$ 間に $1$ 単位の需要が移動するときにかかる
輸送費用を $c_{ij}$，工場 $j$ の容量 $M_j$ とする．
また， $x_{ij}$ を工場 $j$ から顧客 $i$ に輸送される量を表す連続変数する．

上の記号および変数を用いると，輸送問題は以下の線形最適化問題として定式化できる．

$$
\begin{array}{l l l}
minimize  & \displaystyle\sum_{i \in I} \displaystyle\sum_{j \in J} c_{ij} x_{ij}  &     \\
s.t.     &
 \displaystyle\sum_{j \in J} x_{ij} =d_i &  \forall  i \in I  \\
   & \displaystyle\sum_{i \in I} x_{ij} \leq M_j &  \forall  j \in J \\
                 & x_{ij} \geq 0 & \forall  i \in I, j \in J
\end{array}
$$

目的関数は輸送費用の和の最小化であり，最初の制約は需要を満たす条件，
2番目の制約は工場の容量制約である．

```{.python.marimo}
def generate_transportation(n:int =5, m: int=2, seed:int =123) -> (Model, gp.Model):
    random.seed(seed)
    m = m
    n = n
    model = Model(name="Transportation")
    dummy_product = model.addProduct(name="dummy product")
    supplier, customer = {}, {} 
    arc = {}
    activity, mode ={}, {}
    for i in range(m):
        supplier[i] = model.addNode(name=f"Supplier({i})")
        model.addData(dtype="supply", product=dummy_product, node=supplier[i], under_penalty= 0, amount= 100 )
    for j in range(n):
        customer[j] = model.addNode(name=f"Customer({j})")
        model.addData(dtype="demand", product=dummy_product, node=customer[j], amount= 20 )
    for i in range(m):
        for j in range(n):
            arc[i,j] = model.addArc(name=f"arc({i},{j})", source=supplier[i], sink=customer[j])
            activity[i,j] = model.addActivity(name=f"act({i},{j})", atype = "transport", product = dummy_product )
            activity[i,j].addMode ( Mode(name=f"mode({i},{j})", variable_cost = random.randint(5,10) ) )
            arc[i,j].addActivity( activity[i,j] )
    #pprint(model.model_dump_json(exclude=None))
    gp_model = optimize(model)
    return model, gp_model
```

```{.python.marimo}
generate_transport_button = mo.ui.run_button(label="Generate and Optimize Transportation Problem")
_form = mo.vstack(
    [
        # mo.hstack(
        #     [
        #         mo.md("Number of suppliers"),
        #         mo.ui.slider(start=1, stop=10, value=2, show_value=True)
        #     ],
        #     justify="center",
        # ),
        # mo.hstack(
        #     [
        #         mo.md("Number of customers"),
        #         mo.ui.slider(start=1, stop=10, value=5, show_value=True)
        #     ],
        #     justify="center",
        # ),
        mo.hstack(
            [
                generate_transport_button
            ],
            justify="center",
        )
    ]
)
_form
```

```{.python.marimo}
mo.stop(
    not generate_transport_button.value,
    mo.md(text="輸送問題を表示するには上のボタンを押してください。"),
)
_model, _gp_model = generate_transportation()
_g, _bom = visualize(_model, size=20)
# Marimo で表示
mo.image(_g.pipe(format='png'))
```

```{.python.marimo}
# random.seed(123)
# _m = 2
# _n = 5
# _model = Model(name="Transportation")
# _dummy_product = _model.addProduct(name="dummy product")
# _supplier, _customer = ({}, {})
# _arc = {}
# _activity, _mode = ({}, {})
# for _i in range(_m):
#     _supplier[_i] = _model.addNode(name=f"Supplier({_i})")
#     _model.addData(
#         dtype="supply",
#         product=_dummy_product,
#         node=_supplier[_i],
#         under_penalty=0,
#         amount=100,
#     )
# for _j in range(_n):
#     _customer[_j] = _model.addNode(name=f"Customer({_j})")
#     _model.addData(
#         dtype="demand", product=_dummy_product, node=_customer[_j], amount=20
#     )
# for _i in range(_m):
#     for _j in range(_n):
#         _arc[_i, _j] = _model.addArc(
#             name=f"arc({_i},{_j})", source=_supplier[_i], sink=_customer[_j]
#         )
#         _activity[_i, _j] = _model.addActivity(
#             name=f"act({_i},{_j})", atype="transport", product=_dummy_product
#         )
#         _activity[_i, _j].addMode(
#             Mode(name=f"mode({_i},{_j})", variable_cost=random.randint(5, 10))
#         )
#         _arc[_i, _j].addActivity(_activity[_i, _j])
# _gp_model = optimize(_model)
```

## Example: Facility Location


**容量制約付き施設配置問題**(capacitated facility location problem)は，以下のように定義される問題である．

顧客数を $n$， 施設数を $m$ とし，
顧客を $i=1,2,\ldots,n$，
施設を $j=1,2,\ldots,m$ と番号で表すものとする．
また，顧客の集合を $I=\{1,2,\ldots,n\}$，施設の集合を $J=\{1,2,\ldots,m\}$ と記す．

顧客 $i$ の需要量を $d_i$，顧客 $i$ と施設 $j$ 間に $1$ 単位の需要が移動するときにかかる
輸送費用を $c_{ij}$，施設 $j$ を開設するときにかかる固定費用を $f_j$，容量を $M_j$ とする．

以下に定義される連続変数 $x_{ij}$ および $0$-$1$ 整数変数 $y_j$ を用いる．

$$
  x_{ij}= 顧客 i の需要が施設 j によって満たされる量
$$

$$
  y_j = \left\{
           \begin{array}{ll}
               1  & 施設 j を開設するとき \\
               0  & それ以外のとき
           \end{array}
         \right.
$$

上の記号および変数を用いると，容量制約付き施設配置問題は以下の混合整数最適化問題
として定式化できる．

$$
\begin{array}{l l l }
 minimize  & \sum_{j \in J} f_j y_j +\sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} &     \\
 s.t. &  \sum_{j \in J} x_{ij} =d_i  &  \forall  i \in I \\
                 & \sum_{i \in I} x_{ij} \leq M_j y_j & \forall j \in J  \\
                 &  x_{ij} \leq d_i y_j & \forall  i \in I; j \in J \\
                 & x_{ij} \geq 0    & \forall  i \in I; j \in J \\
                 & y_j \in \{ 0,1 \} & \forall  j \in J
\end{array}
$$

```{.python.marimo}
def generate_flp(n:int =5, m: int=2, seed:int =123) -> (Model, gp.Model):
    random.seed(seed)
    m = m
    n = n

    model = Model(name="Facility Location")
    dummy_product = model.addProduct(name="dummy product")
    supplier, customer = {}, {} 
    arc = {}
    activity, mode ={}, {}
    for i in range(m):
        supplier[i] = model.addNode(name=f"Supplier({i})")
        # model.addData(dtype="supply", product=dummy_product, node=supplier[i], under_penalty= 0, amount= 100 )
    for j in range(n):
        customer[j] = model.addNode(name=f"Customer({j})")
        model.addData(dtype="demand", product=dummy_product, node=customer[j], amount= 20 )
    for i in range(m):
        for j in range(n):
            arc[i,j] = model.addArc(name=f"arc({i},{j})", source=supplier[i], sink=customer[j])
            activity[i,j] = model.addActivity(name=f"act({i},{j})", atype = "transport", product = dummy_product )
            activity[i,j].addMode ( Mode(name=f"mode({i},{j})", variable_cost = random.randint(5,10) ) )
            arc[i,j].addActivity( activity[i,j] )
    #add resource and node activity
    resource = {} 
    for i in range(m):
        resource[i] = model.addResource(name=f"resource({i})", capacity = 100., fixed_cost=100.)
        activity[i] = model.addActivity(name=f"act({i})",atype="make",product=dummy_product)
        mode[i] = Mode(name=f"mode({i})", variable_cost = 1.)
        mode[i].addResource(resource= resource[i], variable = 1.)
        activity[i].addMode(mode[i])
        supplier[i].addActivity(activity[i])

    #pprint(model.model_dump_json(exclude=None))
    gp_model = optimize(model)

    return  model, gp_model
```

```{.python.marimo}
generate_flp_button = mo.ui.run_button(label="Generate and Optimize Facility Location Problem")
generate_flp_button
```

```{.python.marimo}
mo.stop(
    not generate_flp_button.value,
    mo.md(text="施設配置問題を表示するには上のボタンを押してください。"),
)
_model, _gp_model = generate_flp()
_g, _bom = visualize(_model, size=20)
mo.image(_g.pipe(format='png'))
```

```{.python.marimo}
# _m = 2
# _n = 5
# _model = Model(name="Facility Location")
# _dummy_product = _model.addProduct(name="dummy product")
# _supplier, _customer = ({}, {})
# _arc = {}
# _activity, _mode = ({}, {})
# for _i in range(_m):
#     _supplier[_i] = _model.addNode(name=f"Supplier({_i})")
# for _j in range(_n):
#     _customer[_j] = _model.addNode(name=f"Customer({_j})")
#     _model.addData(
#         dtype="demand", product=_dummy_product, node=_customer[_j], amount=20
#     )
# for _i in range(_m):
#     for _j in range(_n):
#         _arc[_i, _j] = _model.addArc(
#             name=f"arc({_i},{_j})",
#             source=_supplier[_i],
#             sink=_customer[_j],
#         )
#         _activity[_i, _j] = _model.addActivity(
#             name=f"act({_i},{_j})", atype="transport", product=_dummy_product
#         )
#         _activity[_i, _j].addMode(
#             Mode(name=f"mode({_i},{_j})", variable_cost=random.randint(5, 10))
#         )
#         _arc[_i, _j].addActivity(_activity[_i, _j])
# _resource = {}
# for _i in range(_m):
#     _resource[_i] = _model.addResource(
#         name=f"resource({_i})", capacity=100.0, fixed_cost=100.0
#     )
#     _activity[_i] = _model.addActivity(
#         name=f"act({_i})", atype="make", product=_dummy_product
#     )
#     _mode[_i] = Mode(name=f"mode({_i})", variable_cost=1.0)
#     _mode[_i].addResource(resource=_resource[_i], variable=1.0)
#     _activity[_i].addMode(_mode[_i])
#     _supplier[_i].addActivity(_activity[_i])
# _gp_model = optimize(_model)
```

```{.python.marimo}
# g, bom = visualize(model, size=20)
# g
```

## Example: Fixed Charge Multicommodity Network Flow


多品種流問題に枝上の固定費用 $F: E \rightarrow \mathbf{R}_+$ をつけた問題を，**多品種ネットワーク設計問題**(multi-commodity network design problem)とよぶ．
この問題は，$NP$-困難であり，しかも以下の通常の定式化だと小規模の問題例しか解けない．実務的には，パス型の定式化や列生成法を用いることが推奨される．

枝を使用するか否かを表す $0$-$1$変数を用いる．

- 目的関数：

$$
  \sum_{k \in K} \sum_{e \in E} c_e^k x_{e}^k +  \sum_{e \in E} F_{ij} y_{ij}
$$

- フロー整合条件：

$$
 \sum_{j: ji \in E} x_{ji}^k - \sum_{j: ij \in E} x_{ij}^k =
\left\{
 \begin{array}{ll}
 -b_k  & i=s_k \\
 0  & \forall i \in V \setminus \{s_k,t_k\} \\
 b_k & i=t_k
 \end{array}
 \right.
 ,\forall k \in K
$$

- 容量制約:

$$
  \sum_{k \in K} x_{ij}^k \leq u_{ij} y_{ij}  \ \ \  \forall (i,j) \in E
$$

- 非負制約:

$$
  x_{e}^k \geq 0  \ \ \  \forall e \in E,  k \in K
$$

- 整数制約：

$$
  y_{ij}  \in \{ 0,1 \}   \ \ \  \forall (i,j) \in E
$$

```{.python.marimo}
def generate_fcndp(n:int =5, m: int=2, seed:int =123) -> (Model, gp.Model):
    random.seed(seed)
    m = m
    n = n

    cost_lb, cost_ub = 10, 10
    cap_lb, cap_ub = 150, 150
    demand_lb, demand_ub = 10, 30
    G = nx.grid_2d_graph(m, n)
    D = G.to_directed()
    for (i, j) in D.edges():
        D[i][j]["cost"] = random.randint(cost_lb, cost_ub)
        D[i][j]["capacity"] = random.randint(cap_lb, cap_ub)
    pos = {(i, j): (i, j) for (i, j) in G.nodes()}
    b = {}
    K = []
    for i in D.nodes():
        for j in D.nodes():
            if i != j:
                K.append((i, j))
                b[i, j] = random.randint(demand_lb, demand_ub)

    model = Model(name="Multicommodity Flow")
    product = {}
    node, arc = {}, {}
    activity, mode ={}, {}
    resource = {}
    #node
    for i in D.nodes():
        node[i] = model.addNode(name=f"node({i})", location=pos[i])
    #products
    for (i,j) in b:
        product[i,j] = model.addProduct(name=f"commodity{i},{j}")
        model.addData(dtype="demand", product=product[i,j], node=node[j], amount=b[i,j])
        model.addData(dtype="supply", product=product[i,j], node=node[i], amount=b[i,j])
    #activity
    for (i, j) in D.edges():
        arc[i,j] = model.addArc(name=f"arc({i},{j})", source=node[i], sink=node[j])
        resource[i,j] = model.addResource(name=f"resource({i},{j})", capacity=D[i][j]["capacity"], fixed_cost=100.)
        for (k,l) in product:
            activity[i,j,k,l] = model.addActivity(name=f"act({i},{j}.{k}.{l})", atype = "transport", product = product[k,l] )
            mode[i,j,k,l] =  Mode(name=f"mode({i},{j},{k},{l})", variable_cost = D[i][j]["cost"] )
            mode[i,j,k,l].addResource(resource=resource[i,j], variable=1.)
            activity[i,j,k,l].addMode ( mode[i,j,k,l] )
            arc[i,j].addActivity( activity[i,j,k,l] )

    model.update()
    gp_model = optimize(model)
    return model, gp_model
```

```{.python.marimo}
generate_fcmcp_button = mo.ui.run_button(label="Generate and Optimize Fixed Charge Network Design Problem")
generate_fcmcp_button
```

```{.python.marimo}
mo.stop(
    not generate_fcmcp_button.value,
    mo.md(text="固定費用付きネットワーク設計問題を表示するには上のボタンを押してください。"),
)
_model, _gp_model = generate_fcndp()
_g, _bom = visualize(_model, rankdir="TB", size=60)
#print(_g)
#mo.image(_g.pipe(format='png')) #too large!
# _model.model_dump_json(exclude_defaults=True)
visualize_network(_model)
plt.gca()
```

## Example: Logistics Network Design

多期間複数ラインのロジスティクス・ネットワーク設計モデルの例題である。某飲料メーカーの事例をもとにしている。

```{.python.marimo}
def generate_lndp(seed:int =123) -> (Model, gp.Model):
    random.seed(seed)    
    model = Model(name="multi line lndp")
    T = 3 #planning horizon
    NumProds = 2
    NumPlants = 2
    NumCusts = 2
    NumLines = 2
    NumAreas = 2
    model.interest_rate =0.15
    area = {0: {(0, 0), (0, 1), (1, 0)},
            1: {(1, 1)} 
           }

    periods, products, resources, arcs = {}, {}, {}, {}
    activities, modes = {}, {}

    #period
    for t in range(T):
        periods[t] = model.addPeriod(name=t) 
    #product
    for p in range(NumProds):
        products[p] = model.addProduct(name=f"prod({p})")
    #arc
    for i in range(NumPlants):
        for j in range(NumCusts):
            source = Node(name=f"plant({i})")
            sink = Node(name=f"customer({j})")
            arcs[i,j] = model.addArc(name=f"arc({i},{j})", source=source, sink=sink  )

    #resource(production)
    for i in range(NumPlants):
        for m in range(NumLines):
            resources[i,m] = model.addResource( name=f"resource({i},{m})", capacity =100. )
    #resource(Area: generic transport)
    for r in range(NumAreas):
        resources[r] = model.addResource( name=f"area({r})", capacity =300. )

    #make activity and mode
    for i in range(NumPlants):
        for m in range(NumLines):
            modes[i,m] = model.addMode(name=f"mode({i},{m})")
            modes[i,m].addResource(resource=resources[i,m], fixed=3., variable=1.) #資源使用量
            modes[i,m].addCost(fixed =100., variable=3.) #段取り費用と生産変動費用

    for i in range(NumPlants):
        for p in range(NumProds):
            activities[i,p] = model.addActivity(name=f"act({i},{p})", atype="make", product=products[p] ) 
            for m in range(NumLines):
                activities[i,p].addMode(modes[i,m])                    
            model.nodes[f"plant({i})"].addActivity( activities[i,p] )  

    #transport activity
    for (i,j) in arcs:
        for p in range(NumProds):
            activities[i,j,p] = model.addActivity(name=f"act({i},{j},{p})", atype="transport", product=products[p] ) 
            modes[i,j,p] = model.addMode(name=f"trans_mode({i},{j},{p})")
            modes[i,j,p].addCost(variable=1.) #輸送費用
            #Area制約
            if (i,j) in area[0]:
                modes[i,j,p].addResource(resource = resources[0], variable=4.)
            else:
                modes[i,j,p].addResource(resource = resources[1], variable=5.)

            activities[i,j,p].addMode(mode=modes[i,j,p])
            arcs[i,j].addActivity(activities[i,j,p])

    #demand
    for j in range(NumCusts):
        for p in range(NumProds):
            for t in periods:
                model.addData(dtype="demand", product=products[p],node=model.nodes[f"customer({j})"],period=periods[t],amount=5.)

    #value
    for j in range(NumCusts):
        for p in range(NumProds):
            model.addData(dtype="value", product=products[p],node=model.nodes[f"customer({j})"], amount=10.)
    for j in range(NumPlants):
        for p in range(NumProds):
            model.addData(dtype="value", product=products[p],node=model.nodes[f"plant({j})"], amount=1.)

    model.update()
    gp_model = optimize(model, start_period_name=0, end_period_name=2)

    return model, gp_model
```

```{.python.marimo}
generate_lndp_button = mo.ui.run_button(label="Generate and Optimize Logistics Network Design Problem")
generate_lndp_button
```

```{.python.marimo}
mo.stop(
    not generate_lndp_button.value,
    mo.md(text="ロジスティクス・ネットワーク設計問題を表示するには上のボタンを押してください。"),
)
_model, _gp_model = generate_lndp()
_g, _bom = visualize(_model, size=20)
mo.image(_g.pipe(format='png'))
```

```{.python.marimo}
# f = open('model.json', 'w')
# f.write(model.model_dump_json(exclude_none=True))
# f.close()
```

```{.python.marimo}
# g, bom = visualize(model, size=100)
# g
```

### JSONの可視化

JSON VIsualizer https://omute.net/ を使う。
<!---->
### Optimization

cost (Node,Arc,Mode)にfixed,variable,piecewiseを定義


変数

- $x_{imt}^p$: 点 $i$，モード $m$，期 $t$ における製品 $p$ の生産量
- $y_{imt}^p$: 点 $i$，モード $m$，期 $t$ における製品 $p$ の段取りを表す $0$-$1$ 変数
- $I_{it}^p$: 点 $i$，期 $t$ における製品 $p$ の在庫量
- $X_{ijmt}^p$: 枝 $(i,j)$，モード $m$，期 $t$ における製品 $p$ の輸送量
- $Y_{ijmt}^p$: 枝 $(i,j)$，モード $m$，期 $t$ における製品 $p$ の輸送の有無を表す $0$-$1$ 変数

制約

- フロー整合

$$
I_{i,t-1}^p + \sum_{m} x_{imt}^p = \sum_{j,m} X_{ijmt}^p + I_{i,t}^p  \ \ \ \forall i,p,t
$$

$$
I_{j,t-1}^p + \sum_{i,m} X_{ijmt}^p = \sum_{j,m} d_{jt}^p + I_{i,t}^p  \ \ \ \forall j,p,t
$$

- 資源量上限

$$
\sum_{m,p} varriable_{im} x_{imt}^p + \sum_{m,p} setup_{im} y_{imt}^p \leq CAP_{irt}  \ \ \ \forall i,r,t
$$

$$
\sum_{m,p} varriable_{ijm} X_{imt}^p + \sum_{m,p} setup_{ijm} Y_{imt}^p \leq CAP_{ijrt}  \ \ \ \forall i,j,r,t
$$

- 繋ぎ

$$
x_{imt}^p \leq M y_{imt}^p   \ \ \ \forall i,m,t,p
$$

$$
X_{ijmt}^p \leq M Y_{ijmt}^p   \ \ \ \forall i,j,m,t,p
$$

TODO: 一般化

```{.python.marimo}
def solve_lndp():
    gp_model = gp.Model()
    x, y = ({}, {})
    X, Y = ({}, {})
    fc, vc = ({}, {})
    FC, VC = ({}, {})
    I = {}
    for _i, _node in model.nodes.items():
        if _node.activities is not None and len(_node.activities) > 0:
            for aname, _act in _node.activities.items():
                if _act.atype == "make":
                    _p = act.product.name
                    for _t in model_3.periods:
                        I[i, p, t] = gp_model_4.addVar(
                            name=f"I({i},,{p},{t})", vtype="C"
                        )
                        I[i, p, -1] = 0.0
                    for _m, _mode in _act.modes.items():
                        for _t in model_3.periods:
                            x[i, m, p, t] = gp_model_4.addVar(
                                name=f"x({i},{m},{p},{t})", vtype="C"
                            )
                            vc[i, m, p, t] = mode_2.variable_cost
                            if (
                                _mode.fixed_cost is not None
                                and _mode.fixed_cost > 0.0
                            ):
                                y[i, m, p, t] = gp_model_4.addVar(
                                    name=f"y({i},{m},{p},{t})", vtype="B"
                                )
                                fc[i, m, p, t] = mode_2.fixed_cost
    for (_i, j_4), _arc in model_3.arcs.items():
        if _arc.activities is not None and len(_arc.activities) > 0:
            for aname, _act in _arc.activities.items():
                if _act.atype == "transport":
                    _p = act.product.name
                    for _m, _mode in _act.modes.items():
                        for _t in model_3.periods:
                            X[i, j_4, m, p, t] = gp_model_4.addVar(
                                name=f"x({i},{j_4},{m},{p},{t})", vtype="C"
                            )
                            VC[i, j_4, m, p, t] = mode_2.variable_cost
                            if (
                                _mode.fixed_cost is not None
                                and _mode.fixed_cost > 0.0
                            ):
                                Y[i, j_4, m, p, t] = gp_model_4.addVar(
                                    name=f"Y({i},{j_4},{m},{p},{t})", vtype="B"
                                )
                                FC[i, j_4, m, p, t] = mode_2.fixed_cost
    if GUROBI:
        model_3.update()
    Xtl = gp.tuplelist(list(X.keys()))
    for _i, _node in model_3.nodes.items():
        if _node.activities is not None and len(_node.activities) > 0:
            for aname, _act in _node.activities.items():
                if _act.atype == "make":
                    _p = act.product.name
                    for _t in model_3.periods:
                        gp_model_4.addConstr(
                            I[_i, _p, _t - 1]
                            + gp.quicksum(
                                (x[_i, _m, _p, _t] for _m in _act.modes)
                            )
                            - gp.quicksum(
                                (
                                    X[_i, j, _m, _p, _t]
                                    for _i, j, _m, _p, _t in Xtl.select(
                                        _i, "*", "*", _p, _t
                                    )
                                )
                            )
                            - I[_i, _p, _t]
                            == 0.0,
                            name=f"flow({_i},{_p},{_t})",
                        )
    demands = {}
    for (_dtype, _p, j_4, _t), _data in model_3.data.items():
        if _dtype == "demand":
            demands[p, j_4, t] = data.amount
    for (_p, j_4, _t), demand in demands.items():
        gp_model_4.addConstr(
            I[_i, _p, _t - 1]
            + gp.quicksum(
                (
                    X[_i, j, _m, _p, _t]
                    for _i, j, _m, _p, _t in Xtl.select("*", j_4, "*", _p, _t)
                )
            )
            - I[_i, _p, _t]
            == demand,
            name=f"demand({_p},{j_4},{_t})",
        )
    for _r, _resource in model_3.resources.items():
        node_indices = defaultdict(list)
        arc_indices = defaultdict(list)
        for _m, amount in _resource.modes.items():
            _mode = model_3.modes[m]
            for _a in _mode.activities:
                _act = model_3.activities[a]
                _p = act.product.name
                if _act.nodes is not None:
                    for _i in _act.nodes:
                        node_indices[_i, _m].append(_p)
                elif _act.arcs is not None:
                    for _i, j_4 in _act.arcs:
                        arc_indices["all"].append((_i, j_4, _m, _p))
        if len(node_indices[_i, _m]) > 0:
            for _t in model_3.periods:
                for (_i, _m), prod_list in node_indices.items():
                    gp_model_4.addConstr(
                        gp.quicksum(
                            (
                                amount[1] * x[_i, _m, _p, _t]
                                + amount[0] * _y[_i, _m, _p, _t]
                                for _p in prod_list
                            )
                        )
                        <= _resource.capacity,
                        name=f"capacity({_i},{_r},{_t})",
                    )
        elif len(arc_indices["all"]) > 0:
            for _t in model_3.periods:
                gp_model_4.addConstr(
                    gp.quicksum(
                        (
                            amount[1] * X[_i, j, _m, _p, _t]
                            for _i, j, _m, _p in arc_indices["all"]
                        )
                    )
                    <= _resource.capacity,
                    name=f"area_capacity({_r},{_t})",
                )
    for _i, _node in model_3.nodes.items():
        if _node.activities is not None and len(_node.activities) > 0:
            for aname, _act in _node.activities.items():
                if _act.atype == "make":
                    _p = act.product.name
                    for _m, _mode in _act.modes.items():
                        for _t in model_3.periods:
                            gp_model_4.addConstr(
                                x[_i, _m, _p, _t] <= 99999 * _y[_i, _m, _p, _t]
                            )
    h = {}
    for (_dtype, _p, _i, _), _data in model_3.data.items():
        if _dtype == "value":
            h[p, i] = data.amount * model_3.interest_rate
    gp_model_4.setObjective(
        gp.quicksum(
            (vc[_i, _m, _p, _t] * x[_i, _m, _p, _t] for _i, _m, _p, _t in x)
        )
        + gp.quicksum(
            (fc[_i, _m, _p, _t] * _y[_i, _m, _p, _t] for _i, _m, _p, _t in _y)
        )
        + gp.quicksum(
            (
                VC[_i, j, _m, _p, _t] * X[_i, j, _m, _p, _t]
                for _i, j, _m, _p, _t in X
            )
        )
        + gp.quicksum((h[_p, _i] * I[_i, _p, _t] for _i, _p, _t in I)),
        gp.GRB.MINIMIZE,
    )
    gp_model_4.optimize()
    print("Obj. Val=", gp_model_4.ObjVal)
    for _i, _m, _p, _t in x:
        if x[_i, _m, _p, _t].X > 0:
            print(_i, _m, _t, _p, x[_i, _m, _p, _t].X)
    for _i, j_5, _m, _p, _t in X:
        if X[_i, j_5, _m, _p, _t].X > 0:
            print(_i, j_5, _m, _t, _p, X[_i, j_5, _m, _p, _t].X)
    for _i, _p, _t in I:
        if _t >= 0 and I[_i, _p, _t].X > 0:
            print(_i, _p, _t, I[_i, _p, _t].X)
```

## Example: Multi-Echelon Lotsizing

ロットサイズ決定モデルの例題である。某化学メーカーの事例をもとにしている。

**集合:**

-  $\{1..T\}$: 期間の集合
-  $P$ : 品目の集合（完成品と部品や原材料を合わせたものを「品目」と定義する）
-  $K$ : 生産を行うのに必要な資源（機械，生産ライン，工程などを表す）の集合
-  $P_k$ : 資源 $k$ で生産される品目の集合
-  $Parent_p$ : 部品展開表における品目（部品または材料）$p$ の親品目の集合．言い換えれば，品目 $p$ から製造される品目の集合

**パラメータ:**

-  $T$: 計画期間数．期を表す添え字を $1,2,\ldots,t,\ldots,T$ と記す
-  $f_t^p$ : 期 $t$ に品目 $p$ に対する段取り替え（生産準備）を行うときの費用（段取り費用）
-  $g_t^p$ : 期 $t$ に品目 $p$ に対する段取り替え（生産準備）を行うときの時間（段取り時間）
-  $c_t^p$ : 期 $t$ における品目 $p$ の生産変動費用
-  $h_t^p$ : 期 $t$ から期 $t+1$ に品目 $p$ を持ち越すときの単位あたりの在庫費用
-  $d_t^p$ : 期 $t$ における品目 $p$ の需要量
-  $\phi_{pq}$ : $q \in Parent_p$ のとき， 品目 $q$ を $1$ 単位製造するのに必要な品目 $p$ の数 （$p$-units）．
ここで， $p$-unitsとは，品目 $q$ の $1$単位と混同しないために導入された単位であり， 品目 $p$ の $1$単位を表す．$\phi_{pq}$ は，部品展開表を有向グラフ表現したときには，枝の重みを表す
-  $M_t^k$ : 期 $t$ における資源 $k$ の使用可能な生産時間の上限． 定式化では，品目 $1$単位の生産時間を $1$単位時間になるようにスケーリングしてあるものと仮定しているが， プログラム内では単位生産量あたりの生産時間を定義している
-  $UB_t^p$ : 期 $t$ における品目 $p$ の生産時間の上限
   品目 $p$ を生産する資源が $k$ のとき，資源の使用可能時間の上限 $M_t^k$ から段取り替え時間 $g_t^p$ を減じたものと定義される

**変数:**

-  $x_t^p$（x）: 期 $t$ における品目 $p$ の生産量
-  $I_t^p$（inv） : 期 $t$ における品目 $p$ の在庫量
-  $y_t^p$（y）: 期 $t$ に品目 $p$ に対する段取りを行うとき $1$， それ以外のとき $0$ を表す $0$-$1$ 変数


上の記号を用いると、多段階ロットサイズ決定モデルは，以下のように定式化できる．

$$
\begin{array}{ l l l }
minimize & \sum_{t=1}^T \sum_{p \in P} \left( f_t^p y_t^p + c_t^p x_t^p + h_t^p I_t^p \right) &           \\
s.t. &  \ \ I_{t-1}^p +x_t^p  = d_t^p+ \sum_{q \in Parent_p} \phi_{pq} x_t^q  +I_t^p & \forall p \in P, t=1,\ldots,T  \\
            &  \sum_{p \in P_k} x_t^p  +\sum_{p \in P_k} g_t^p y_t^p \leq M_t^k    & \forall k \in K, t=1,\ldots,T \\
            &  x_t^p  \leq UB_t^p y_t^p     & \forall p \in P, t=1,\ldots,T  \\
            &  I_0^p =0               & \forall p \in P                          \\
            &  x_t^p,I_t^p \geq 0         & \forall  p \in P, t=1,\ldots,T  \\
            &  y_t \in \{0,1\}   & \forall t=1,\ldots,T
\end{array}
$$


上の定式化で，最初の制約式は，各期および各品目に対する在庫の保存式を表す．
より具体的には，品目 $p$ の期 $t-1$ からの在庫量 $I_{t-1}^p$ と生産量 $x_t^p$ を加えたものが，
期 $t$ における需要量 $d_t^p$，次期への在庫量 $I_t^p$，
および他の品目を生産するときに必要な量 $\sum_{q \in Parent_p} \phi_{pq} x_t^q$ の和に等しいことを表す．

2番目の制約は， 各期の生産時間の上限制約を表す． 定式化ではすべての品目の生産時間は，
$1$ 単位あたり$1$ 時間になるようにスケーリングしてあると仮定していたが，実際問題のモデル化の際には，
品目 p を $1$ 単位生産されるときに，資源 $r$ を使用する時間を用いた方が汎用性がある．

3番目の式は，段取り替えをしない期は生産できないことを表す．

```{.python.marimo}
def generate_lotsize():

    prod_df = pd.read_csv(folder + "lotprod.csv", index_col=0)
    prod_df.set_index("name", inplace=True)
    production_df = pd.read_csv(folder + "production.csv", index_col=0)
    bom_df = pd.read_csv(folder + "bomodel.csv", index_col=0)
    resource_df = pd.read_csv(folder + "resource.csv", index_col=0)
    plnt_demand_df = pd.read_csv(folder + "plnt-demand.csv", index_col=0)
    demand_df = pd.pivot_table(
        plnt_demand_df,
        index="prod",
        columns="period",
        values="demand",
        aggfunc="sum",
    )

    raw_materials = set(bom_df["child"])
    final_products = set(bom_df["parent"])
    items = raw_materials | final_products

    model = Model(name="lot sizing model")
    model.interest_rate = 0.15
    T = len(demand_df.columns)
    periods, products, resources, arcs = {}, {}, {}, {}
    activities, modes = {}, {}
    #period
    for t in range(T):
        periods[t] = model.addPeriod(name=t) 
    #products
    for row in prod_df.itertuples():
        p = str(row.Index)
        products[p] = model.addProduct(name=f"prod({p})", value=row.inv_cost/model.interest_rate)
    #demand
    demand_matrix = demand_df.values
    for i,p in enumerate(products):
        if i>=10:
            break #部品の需要はなし
        for t in periods:
            model.addData(dtype="demand",product=products[p],period=periods[t],amount= float(demand_matrix[i,t]))
    #resource 期ごとに同じ容量を仮定（２つの資源があり，Res0は原料生産，Res1は最終製品生産に用いられる）
    for row in resource_df.itertuples():
        resources[row.name] = model.addResource(name=row.name, capacity=row.capacity)
    #activity and mode
    for row in production_df.itertuples():
        p = row.name
        modes[p] = model.addMode(name=p, fixed_cost=row.SetupCost, variable_cost=row.ProdCost)
        if p in raw_materials:
            modes[p].addResource(resources["Res0"], fixed=row.SetupTime, variable = row.ProdTime)
        elif p in final_products:
            modes[p].addResource(resources["Res1"], fixed=row.SetupTime, variable = row.ProdTime)
        activities[p] = model.addActivity(name=p, atype="make", product=products[p])
        activities[p].addMode( modes[p] )
    #bom
    for row in bom_df.itertuples():
        child = row.child
        p = row.parent
        modes[p].addComponent(products[child])

    #データの準備
    parent = defaultdict(set) #子品目pを必要とする親品目とモードの組の集合
    phi = defaultdict(float) #親品目qをモードmで１単位生産するために必要な子品目pのunit数
    setup_time = defaultdict(float) #setup_time[p,m,r]
    prod_time = defaultdict(float)
    setup_cost = defaultdict(float)
    prod_cost = defaultdict(float) #prod_cost[p,m]
    demand = defaultdict(float) #demand[t,p]
    item_modes = defaultdict(set) #資源rを使う品目とモードの組の集合

    for (dtype,p,_,t), data in model.data.items():
        if dtype=="demand":
            demand[t,p] = data.amount

    for a, act in model.activities.items():
        product = act.product
        p = product.name
        for m, mode in act.modes.items():
            prod_cost[p,m] = mode.variable_cost
            setup_cost[p,m] = mode.fixed_cost
            for r, requirement in mode.fixed_requirement.items():
                setup_time[p,m,r] = requirement
                item_modes[r].add( (p,m) )
            for r, requirement in mode.variable_requirement.items():
                prod_time[p,m,r] = requirement
            #原材料(components)からparentとphiを生成
            if mode.components is not None:
                for child, quantity in mode.components.items():
                    parent[child].add( (p,m) )
                    phi[child, p, m] = quantity


    gp_model = gp.Model()
    x, I, y = {}, {}, {}
    slack, surplus = {}, {}
    T = len(model.periods)
    Ts = range(0, T)

    for a, act in model.activities.items():
        product = act.product
        p = product.name
        for t in range(T):
            slack[t, p] = gp_model.addVar(name=f"slack({p},{t})")
            surplus[t, p] = gp_model.addVar(name=f"surplus({p},{t})")
            I[t, p] = gp_model.addVar(name=f"I({p},{t})")
            I[-1, p] = 0. #初期在庫（本来ならInventoryクラスで定義）
            I[T-1,p] = 0. #最終在庫
        for m, mode in act.modes.items():
            for t in range(T):
                x[t, m, p] = gp_model.addVar(name=f"x({p},{m},{t})")
                y[t, m, p] = gp_model.addVar(name=f"y({p},{m},{t})", vtype="B")

     #各費用項目を別途合計する
    cost ={}
    for i in range(5):
        cost[i] = gp_model.addVar(vtype="C",name=f"cost[{i}]")

    if GUROBI: gp_model.update()           

    for r, resource in model.resources.items():
        for t in Ts:
            # time capacity constraints
            gp_model.addConstr(gp.quicksum(prod_time[p,m,r]*x[t,m,p] + setup_time[p,m,r]*y[t,m,p] for (p,m) in item_modes[r])
                            <= resource.capacity, 
                            f"TimeConstraint1({r},{t})")
    for t in Ts:
        for a, act in model.activities.items():
            product = act.product
            p = product.name
            # flow conservation constraints（ソフト制約）
            gp_model.addConstr(I[t-1, p] + gp.quicksum(x[t, m, p] for m in act.modes) + slack[t, p] - surplus[t, p] - I[t, p] 
                               - gp.quicksum( phi[p,q,m]*x[t, m, q] for (q,m) in parent[p]) == demand[t,p], f"FlowCons({t},{p})" ) 

    for t in Ts:
        for (p,m,r) in setup_time:
            gp_model.addConstr(prod_time[p,m,r]*x[t,m,p]
                        <=  (model.resources[r].capacity -setup_time[p,m,r])*y[t,m,p], f"ConstrUB({t},{m},{r},{p})")

    gp_model.addConstr( gp.quicksum( slack[t, p]+surplus[t, p] for (t,p) in slack) -cost[0] == 0. )
    gp_model.addConstr( gp.quicksum( setup_cost[p,m]*y[t,m,p] for  (t,m,p) in y)-cost[1] == 0.)
    gp_model.addConstr( gp.quicksum( prod_cost[p,m]*x[t,m,p] for (t,m,p) in x) - cost[2] == 0.)
    gp_model.addConstr( gp.quicksum( model.products[p].value*model.interest_rate*I[t, p] for (t,p) in I) - cost[3] == 0. )

    gp_model.setObjective(99999.*cost[0] + gp.quicksum(cost[i] for i in range(2,4)) , gp.GRB.MINIMIZE)
    gp_model.Params.TimeLimit =60
    gp_model.optimize()

    print("Obj. Val.=", gp_model.ObjVal)
    for _i in range(5):
        print(cost[_i].X)

    for _t, _p in I:
        if _t < 0 or _t == T - 1:
            continue
        if I[_t, _p].X > 0:
            print(_t, _p, I[_t, _p].X)

    return model, gp_model
```

```{.python.marimo}
generate_lotsize_button = mo.ui.run_button(label="Generate and Optimize Lotsizing Problem")
generate_lotsize_button
```

```{.python.marimo}
mo.stop(
    not generate_lotsize_button.value,
    mo.md(text="多段階ロットサイズ決定問題を表示するには上のボタンを押してください。"),
)
_model, _gp_model = generate_lotsize()
_g, _bom = visualize(_model, size=20)
mo.image(_g.pipe(format='png'))
```

```{.python.marimo}
# f = open('model2.json', 'w')
# f.write(model.model_dump_json(exclude_none=True))
# f.close()
# g, bom = visualize(model)
# bom
```

<!-- ![A Lot-sizing Model in JSON](../figure/model2.png) -->
<!---->
### Optimization

リード時間 $0$ を仮定

変数

- $x_{mt}^p$: モード $m$，期 $t$ における製品 $p$ の生産量
- $y_{mt}^p$: モード $m$，期 $t$ における製品 $p$ の段取りを表す $0$-$1$ 変数
- $I_{t}^p$: 期 $t$ における製品 $p$ の在庫量

制約

- フロー整合

$$
I_{t-1}^p + \sum_{m} x_{mt}^p = d_{t}^p + \sum_{q} \phi_{pqm} x_{mt}^q + I_{i,t}^p \ \ \ \forall t,p
$$

- 資源量上限

$$
\sum_{m,p} varriable_{m} x_{mt}^p + \sum_{m,p} setup_{m} y_{mt}^p \leq CAP_{rt} \ \ \ \forall r,t
$$

- 繋ぎ

$$
x_{mt}^p \leq M y_{mt}^p  \ \ \ \forall m,t,p
$$
<!---->
## Example: Shift Scheduling

簡単なシフト最適化モデル

TODO: 複数日の問題への拡張

```{.python.marimo}
def generate_shift():
    model = Model(name="Simple Shift scheduling")

    NumProds = 1

    demands = [1,2,3,4,5,4,3,2,2,1]
    T = len(demands) #planning horizon
    NumStaffs = max(demands) + 1 
    min_hours = 3 #最低でも3時間は働く
    wage = 1000 #時給

    periods, products, resources, arcs = {}, {}, {}, {}
    activities, modes = {}, {}

    #period
    for t in range(T):
        periods[t] = model.addPeriod(name=t) 
    #product（製品もOptionalとすべきか？）
    for p in range(NumProds):
        products[p] = model.addProduct(name=f"prod({p})")
    for t,d in enumerate(demands):
        model.addData(dtype="demand", product=products[0], period=periods[t], amount=d)

    for i in range(NumStaffs):
        activities[i] = model.addActivity(name=f"staff({i})", atype="shift", product=products[0])
        for s in range(T-min_hours): #稼働時間 [s,e] （最後の期も含む）
            for e in range(s+min_hours-1, T):
                mode = Mode(name=f"staff({i},{s},{e})", fixed_cost = ((e-s)+1)*wage )
                for t in range(s,e+1):
                   mode.addPeriod( periods[t] )
                activities[i].addMode( mode )

    model.update() #period_indexなどを更新
    gp_model = gp.Model()
    #variables
    x = {}
    cost = {}
    periods = {}
    demands = {}

    for (dtype,p,i,t) in model.data:
        if dtype=="demand":
            demands[ model.period_index[t] ] = model.data[dtype,p,i,t].amount

    for i, a in model.activities.items():
        for m in a.modes:
            periods[i,m] = a.modes[m].periods.keys()
            cost[i,m] = a.modes[m].fixed_cost
            x[i,m] = gp_model.addVar(name=f"x({i},{m})", vtype="B")
    if GUROBI: gp_model.update()
    gp_model.setObjective(gp.quicksum( cost[i,m]*x[i,m] for (i,m) in x  ) ,gp.GRB.MINIMIZE)
    for i, a in model.activities.items():
        gp_model.addConstr(gp.quicksum( x[i,m] for m in a.modes) <=1)
    for t,d in demands.items():
        gp_model.addConstr( gp.quicksum(x[i,m] for (i,m) in x if t in periods[i,m]) ==d)
    gp_model.optimize()

    return model, gp_model
```

### Optimization

変数

- $x_{am}$: 活動（スタッフ） $a$ がモード（シフト） $m$ を選択したとき $1$

目的関数

$$
minimize \ \ \ \sum_{a,m} cost_{am} x_{am}
$$

制約

- モード（シフト）選択

$$
\sum_{m} x_{am} \leq  1 \ \ \ \forall a
$$

- 需要（必要人数）

$$
\sum_{a,m | t \in period[a,m] } x_{am} \geq demand_{t} \ \ \ \forall t
$$

```{.python.marimo}
generate_shift_button = mo.ui.run_button(label="Generate and Optimize Shift Scheduling Problem")
generate_shift_button
```

```{.python.marimo}
mo.stop(
    not generate_shift_button.value,
    mo.md(text="シフトスケジューリング問題を表示するには上のボタンを押してください。"),
)
_model, _gp_model = generate_shift()
_g, _bom = visualize(_model, size=50)
mo.image(_g.pipe(format='png'))
```

### Multi-day Entension

日をまたぐ制約をConstraintクラスで記述

```{.python.marimo}
#| hide
def generate_multiday_shift():
    model = Model(name="Shift scheduling")

    NumProds = 1

    demands = {0:[1,2,3,4,5,4,3,2,2,1],
               1:[1,2,3,4,5,5,3,2,2,1],
               2:[2,2,3,4,5,5,3,2,1,1]
              }

    NumDays = len(demands) 
    T = len(demands[0])    #one day planning horizon
    NumStaffs = max(demands[0]) + 1 
    min_hours = 3 #最低でも3時間は働く
    wage = 1000 #時給

    periods, products = {}, {}
    activities, modes = {}, {}

    #dummy product
    for p in range(NumProds):
        products[p] = model.addProduct(name=f"prod({p})")

    #period
    for day, dem in demands.items():
        for t in range(len(dem)):
            periods[day,t] = model.addPeriod(name=f"period({day},{t})", start= datetime(year=2024,month=1,day=1+day,hour=8+t) )

    #demand data
    for day, dem in demands.items():
        for t,d in enumerate(dem):
            model.addData(dtype="demand", product=products[0], period=periods[day,t], amount=d)

    for day in range(NumDays):
        for i in range(NumStaffs):
            activities[i,day] = model.addActivity(name=f"staff({i},{day})", atype="shift", product=products[0])
            for s in range(T-min_hours): #稼働時間 [s,e] （最後の期も含む）
                for e in range(s+min_hours-1, T):
                    mode = Mode(name=f"staff({i},{day},{s},{e})", fixed_cost = ((e-s)+1)*wage )
                    for t in range(s,e+1):
                       mode.addPeriod( periods[day,t] )
                    activities[i,day].addMode( mode )

    #日をまたぐ制約
    constraints ={}
    for day in range(NumDays-1):
        for i in range(NumStaffs):
            constraints[day,i] = model.addConstraint(name=f"constraint({day},{i})", direction="<=", rhs=9*wage, weight=10) #2日で..時間以内
            for m_name, mode in activities[i,day].modes.items():
                constraints[day,i].addTerms(mode.fixed_cost, activities[i,day], mode)
            for m_name, mode in activities[i,day+1].modes.items():
                constraints[day,i].addTerms(mode.fixed_cost, activities[i,day+1], mode)

    model.update() #period_indexなどを更新
    gp_model = gp.Model()
    #variables
    x = {}
    cost = {}
    periods = {}
    demands = {}

    for (dtype,p,i,t) in model.data:
        if dtype=="demand":
            demands[ t ] = model.data[dtype,p,i,t].amount

    for i, a in model.activities.items():
        for m in a.modes:
            periods[i,m] = a.modes[m].periods.keys()
            cost[i,m] = a.modes[m].fixed_cost
            x[i,m] = gp_model.addVar(name=f"x({i},{m})", vtype="B")

    #逸脱量を表す変数
    slack, surplus ={},{}
    for con_name, constraint in model.constraints.items():
        slack[con_name] = gp_model.addVar(name=f"slack({con_name})")
        surplus[con_name] = gp_model.addVar(name=f"surplus({con_name})")

    if GUROBI: gp_model.update()
    assign_ub_constr ={}
    for i, a in model.activities.items():
        assign_ub_constr[i] = gp_model.addConstr(gp.quicksum( x[i,m] for m in a.modes) <=1, name= f"Assign UB({i})")

    demand_lb_constr ={}
    for t,dem in demands.items():
        demand_lb_constr[t] = gp_model.addConstr( gp.quicksum(x[i,m] for (i,m) in x if t in periods[i,m]) >=dem, name= f"Demand LB{t})")

    for con_name, constraint in model.constraints.items():
        if constraint.direction in ["=","=="]:
            gp_model.addConstr( gp.quicksum(coeff*x[act.name, mode.name] for (coeff, act, mode) in constraint.terms)
                               +slack[con_name]-surplus[con_name] == constraint.rhs,
                               name=con_name )
        elif constraint.direction in ["<=","<"]:
            gp_model.addConstr( gp.quicksum(coeff*x[act.name, mode.name] for (coeff, act, mode) in constraint.terms)
                               -surplus[con_name]<= constraint.rhs,
                               name=con_name )
        elif constraint.direction in [">=",">"]:
            gp_model.addConstr( gp.quicksum(coeff*x[act.name, mode.name] for (coeff, act, mode) in constraint.terms) 
                               +surplus[con_name] >= constraint.rhs,
                               name=con_name )
    gp_model.setObjective(gp.quicksum( cost[i,m]*x[i,m] for (i,m) in x ) + 
                          gp.quicksum( constraint.weight*(slack[con_name]+surplus[con_name]) 
                                      for con_name, constraint in model.constraints.items() )
                          ,gp.GRB.MINIMIZE)

    #gp_model.write("shift.lp")
    gp_model.optimize()
    print("Obj. Val.=",gp_model.ObjVal)
    num_staffs = [0 for t in range(len(demands))]
    for (i,m) in x:
        if x[i,m].X > 0.001:
            print(i,m,x[i,m].X)
            for t in periods[i,m]:
                num_staffs[model.period_index[t]]+=1

    return model, gp_model
```

```{.python.marimo}
# f = open('model3.json', 'w')
# f.write(model.model_dump_json(exclude_none=True))
# f.close()
# g, bom = visualize(model, "TB", 560)
```

### Optimization

```{.python.marimo}
generate_multiday_shift_button = mo.ui.run_button(label="Generate and Optimize Multi-day Shift Scheduling Problem")
generate_multiday_shift_button
```

```{.python.marimo}
mo.stop(
    not generate_multiday_shift_button.value,
    mo.md(text="複数日シフトスケジューリング問題を表示するには上のボタンを押してください。"),
)
_model, _gp_model = generate_multiday_shift()
#_g, _bom = visualize(_model, "TB", 560)
mo.image(_g.pipe(format='png'))
```

<!-- ### 可視化

![A Shift Scheduling Model in JSON](../figure/model3.png) -->
<!---->
## Example: General Logistics Network Design plus Safety Stock Allocation

枝上に活動を定義した一般化ロジスティクス・ネットワーク設計モデル

可視化の際に枝の太さでフロー量を表現できる．


注：区分的線形関数をSOSで解く

集合：

-  $N$: 点の集合．原料供給地点，工場，倉庫の配置可能地点，顧客群の集合などのすべての地点を総称して点とよぶ．

-  $A$: 枝の集合．
少なくとも1つの製品が移動する可能性のある点の対を枝とよぶ．

-  $Prod$: 製品の集合．
製品は，ロジスティクス・ネットワーク内を流れる「もの」の総称である．

以下に定義する $Child_p$， $Parent_p$ は，製品の集合の部分集合である．

- $Child_p$: 部品展開表における製品 $p$ の子製品の集合．言い換えれば，製品 $p$ を製造するために必要な製品の集合．

- $Parent_p$: 部品展開表における製品 $p$ の親製品の集合．言い換えれば，製品 $p$ を分解することによって生成される製品の集合．

- $Res$: 資源の集合．
製品を処理（移動，組み立て，分解）するための資源の総称．
基本モデルでは枝上で定義される．
たとえば，工場を表す枝における生産ライン（もしくは機械）や
輸送を表す枝における輸送機器（トラック，船，鉄道，飛行機など）が資源の代表的な要素となる．

-  $NodeProd$: 需要もしくは供給が発生する点と製品の $2$つ組の集合．

- $ArcRes$: 枝と資源の可能な $2$つ組の集合．
枝 $a \in A$ 上で資源 $r \in Res$ が利用可能なとき，$(a,r)$ の組が
集合 $ArcRes$ に含まれるものとする．

- $ArcResProd$: 枝と資源と製品の可能な $3$つ組の集合．
枝 $a \in A$ 上の資源 $r \in Res$ で製品 $p \in Prod$ の処理が利用可能なとき， $(a,r,p)$ の組が
集合 $ArcResProd$ に含まれるものとする．

以下に定義する $Assemble$，$Disassemble$ および $Transit$ は $ArcResProd$ の部分集合である．

- $Assemble$: 組み立てを表す枝と資源と製品の可能な $3$ つ組の集合．
枝 $a \in A$ 上の資源 $r \in Res$ で製品 $p \in Prod$ の組み立て処理が利用可能なとき，$(a,r,p)$ の組が
集合 $Assemble$ に含まれるものとする．ここで製品 $p$ の組み立て処理とは，子製品の集合 $Child_p$ を用いて $p$ を
製造することを指す．

- $Disassemble$: 分解を表す枝と資源と製品の可能な $3$ つ組の集合．
枝 $a \in A$ 上の資源 $r \in Res$ で製品 $p \in Prod$ の分解処理が利用可能なとき，$(a,r,p)$ の組が
集合 $Disassemble$ に含まれるものとする．ここで製品 $p$ の分解処理とは，$p$ から親製品の集合 $Parent_p$ を
生成することを指す．

- $Transit$: 移動を表す枝と資源と製品の可能な $3$ つ組の集合．
枝 $a \in A$ 上の資源 $r \in Res$ で製品 $p \in Prod$ が形態を変えずに流れることが可能なとき，
$(a,r,p)$ の組は集合 $Transit$ に含まれるものとする．

- $ResProd$: 資源と製品の可能な $2$ つ組の集合．
集合 $ArcResProd$ から生成される．

- $ArcProd$: 枝と製品の可能な $2$ つ組の集合．
集合 $ArcResProd$ から生成される．

入力データ：


-  $D_i^p$: 点  $i$ における製品 $p$ の需要量（$p$-units $/$ 単位期間）；
負の需要は供給量を表す．ここで，$p$-unit とは，製品 $p$ の $1$ 単位を表す．

-  $DPENALTY_{ip}^{+}$:
  点  $i$ における製品 $p$ の $1$ 単位あたりの需要超過（供給余裕）ペナルティ
（円 $/$ 単位期間・$p$-unit）；通常は小さな値

-  $DPENALTY_{ip}^{-}$:
点  $i$ における製品 $p$ の $1$ 単位あたりの需要不足（供給超過）ペナルティ
（円 $/$ 単位期間・$p$-unit）； 通常は大きな値

-  $AFC_{ij}$: 枝 $(i,j)$ を使用するときに発生する固定費用（円 $/$ 単位期間）

-  $ARFC_{ijr}:$ 枝 $(i,j)$ 上で資源 $r$ を使用するときに発生する固定費用（円 $/$ 単位期間）

-  $ARPVC_{ijr}^p$: 枝 $(i,j)$ 上で資源 $r$ を利用して製品 $p$ を $1$ 単位処理するごとに発生する変動費用
（円 $/$ 単位期間・$p$-unit）

- $\phi_{pq}$ : $q \in Parent_p$ のとき， 品目 $q$ を $1$ 単位生成するのに必要な品目 $p$ の数 （$p$-units）； ここで， $p$-unitsとは，品目 $q$ の $1$単位と混同しないために導入された単位であり， 品目 $p$ の $1$単位を表す．$\phi_{pq}$ は，部品展開表を有向グラフ表現したときには，枝の重みを表す．
この値から以下の$U_{p q}$と$\bar{U}_{p q}$を計算する．

-  $U_{p q}$: 製品 $p$ の $1$ 単位を組み立て処理するために必要な製品 $q \in Child_p$ の量（$q$-units）
-  $\bar{U}_{p q}$: 製品 $p$ の $1$ 単位を分解処理して生成される製品 $q \in Parent_p$ の量（$q$-units）

-  $RUB_r$: 資源 $r$ の利用可能量上限（$r$-units）

<!---
-  $RLB_r$: 資源 $r$ の利用可能量下限（$r$-units）；資源を使用しないときには $0$ だが，使用した場合の最低量を表す．
--->

-  $R_{r}^{p}$: 製品 $p$ の $1$ 単位を（組み立て，分解，移動）処理する際に必要な資源 $r$ の量（$r$-units）；
ここで，$r$-unit とは，資源 $r$ の $1$ 単位を表す．

-  $CT_{ijr}^p$: 枝 $(i,j)$ 上で資源 $r$ を利用して製品 $p$ を処理する際のサイクル時間（単位期間）

-  $LT_{ijr}^p$: 枝 $(i,j)$ 上で資源 $r$ を利用して製品 $p$ を処理する際のリード時間（単位期間）

-  $VAL_{i}^p$: 点  $i$ 上での製品 $p$ の価値（円）

-  $SSR_i^p$: 点 $i$ 上での製品 $p$ の安全在庫係数．（無次元）

-  $VAR_p$: 製品 $p$ の変動比率（$p$-units）；これは，製品ごとの需要の分散と平均の比が一定と仮定したとき，
「需要の分散 $/$ 需要の平均」と定義される値である．

-  $ratio$: 利子率（\% $/$ 単位期間）

-  $EIC_{ij}^p$: 枝 $(i,j)$ 上で資源 $r$ を用いて処理（組み立て，分解，輸送）される
製品 $p$ に対して定義されるエシェロン在庫費用（円 $/$単位期間）；
この値は，以下のように計算される．


$$
 EIC_{ijr}^p =\max\{ ratio \times \left(VAL_{j}^p- \sum_{q \in Child_p} \phi_{qp} VAL_{i}^q \right)/ 100, 0 \} \ \ \  (i,j,r,p) \in Assemble
$$

$$
 EIC_{ijr}^p =\max\{ ratio \times \left(\sum_{q \in Parent_p} \phi_{pq} VAL_{j}^q -VAL_{i}^p \right)/ 100, 0 \} \ \ \  (i,j,r,p) \in Disassemble
$$

$$
 EIC_{ijr}^p =\max\{ ratio \times \left(VAL_{j}^p -VAL_{i}^p\right)/ 100, 0 \} \ \ \  (i,j,r,p) \in Transit
$$

- $CFP_{ijr}$: 枝 $(i,j)$ で資源 $r$ を使用したときの$CO_2$排出量 （g)； 輸送の場合には，資源 $r$ の排出原単位 (g/km) に，枝 $(i,j)$ の距離 (km) を乗　じて計算しておく．
- $CFPV_{ijr}$: 資源 $r$ の使用量（輸送の場合には積載重量）あたりの$CO_2$排出原単位（g $/$ $r$-units)
- $CFPUB$: $CO_2$排出量上限（g）

変数は実数変数と$0$-$1$整数変数を用いる．

まず，実数変数を以下に示す．

-  $w_{ijr}^p (\geq 0)$: 枝 $(i,j)$ で資源 $r$ を利用して製品 $p$ を処理する量を表す実数変数（$p$-units $/$ 単位期間）

-  $v_{ip}^+ (\geq 0)$:  点  $i$ における製品 $p$ の需要の超過量（需要が負のときには供給の超過量）
を表す実数変数（$p$-units $/$ 単位期間）

-  $v_{ip}^- (\geq 0)$:  点  $i$ における製品 $p$ の需要の不足量（需要が負のときには供給の不足量）
を表す実数変数（$p$-units $/$ 単位期間）


$0$-$1$整数変数は，枝および枝上の資源の利用の有無を表現する．

-  $y_{ij} (\in \{0,1\})$:  枝$(i,j)$ を利用するとき $1$，それ以外のとき $0$
-  $z_{ijr} (\in \{0,1\})$: 枝$(i,j)$ 上で資源 $r$ を利用するとき $1$，それ以外のとき $0$

定式化：

$$
\begin{array}{ll}
\text{最小化} & \text{枝固定費用} + \text{枝・資源固定費用} + \\
              & \text{枝・資源・製品変動費用} + \text{供給量超過費用} + \\
              & \text{供給量不足費用} + \text{需要量超過費用} + \text{需要量不足費用} + \\
              & \text{サイクル在庫費用} + \text{安全在庫費用} + \text{需要逸脱ペナルティ} \\
\text{条件} & \text{フロー整合条件} \\
              & \text{資源使用量上限} \\
              & \text{枝と枝上の資源の繋ぎ条件} \\
              & \text{$CO_2$排出量上限制約}
\end{array}
$$

- 目的関数の構成要素

$$
 \text{枝固定費用} =  \sum_{(i,j) \in A}  AFC_{ij} y_{ij}
$$

$$
 \text{枝・資源固定費用} =  \sum_{(i,j,r) \in ArcRes}  ARFC_{ijr} z_{ijr}
$$

$$
 \text{枝・資源・製品変動費用}=  \sum_{(i,j,r,p) \in ArcResProd}  ARPVC_{ijr}^p w_{ijr}^p
$$

$$
 \text{需要量超過費用}= \sum_{(i,p) \in NodeProd %: D_{i}^{p}>0
  } DPENALTY_{ip}^+ v_{ip}^+
$$

$$
 \text{需要量不足費用}= \sum_{(i,p) \in NodeProd %: D_{i}^{p}>0
 } DPENALTY_{ip}^- v_{ip}^-
$$

$$
 \text{サイクル在庫費用} = \sum_{(i,j,r,p) \in ArcResProd} \frac{EIC_{ijr}^p CT_{ijr}^p }{2} w_{ijr}^p
$$

$$
 \text{安全在庫費用} = \sum_{(i,j,r,p) \in ArcResProd}
   ratio \times VAL_j^p SSR_i^p \sqrt{VAR_p LT_{ijr}^p  w_{ijr}^p}
$$

上式における平方根は区分的線形関数で近似するものとする．

$$
 \text{需要逸脱ペナルティ} = \sum_{(i,p) \in NodeProd}  DPENALTY_{ip}^{+} v_{ip}^+  + DPENALTY_{ip}^{-} v_{ip}^-
$$


- 一般化フロー整合条件


$$
  \sum_{r \in Res, j \in N: (j,i,r,p) \in Transit \cup Assemble} w_{jir}^p+
  \sum_{r \in Res, j \in N: (j,i,r,q) \in Disassemble, p \in Parent_q} \phi_{qp} w_{jir}^q \\
  =
   \sum_{r \in Res, k \in N: (i,k,r,p) \in Transit \cup Disassemble} w_{ikr}^p+
 \sum_{r \in Res, k \in N: (i,k,r,q) \in Assemble, p \in Child_q} \phi_{pq} w_{ikr}^q+ \\
(\text{ if  } (i,p) \in NodeProd \text{ then  } D_i^p -v_{ip}^{-} +v_{ip}^{+}
 \text{ else }  0) \ \ \  \forall i \in N, p \in Prod
$$

- 資源使用量上限


$$
\sum_{p \in Prod: (i,j,r,p) \in ArcResProd} R_{r}^p w_{ijr}^p \leq RUB_{r} z_{ijr}
\ \ \  \forall (i,j,r) \in ArcRes
$$

- 枝と枝上の資源の繋ぎ条件


$$
 z_{ijr} \leq y_{ij} \ \ \  \forall (i,j,r) \in ArcRes
$$

- $CO_2$排出量上限制約

$$
\sum_{(i,j,r) \in ArcRes} CFP_{ijr} z_{ijr} +  \sum_{ (i,j,r,p) \in ArcResProd} CFPV_{ijr}  R_r^p w_{ijr}^p \leq CFPUB
$$

```{.python.marimo}
def generate_general_lndp():

    NodeData=[ "source1", "source2", "plantin", "plantout", "customer" ]
    #Arc list, fixed cost (ArcFC) and Diatance (km)
    ArcData={("source1", "plantin"):  [100,200],
            ("source2", "plantin"):  [100,500],
            ("plantin", "plantout"):  [100,0],
            ("plantout", "customer"): [100,20],
          }
    #Prod data
    #weight (ton), variability of product (VAR)= variance/mean
    ProdData={
          "apple": [0.01,0],
          "melon": [0.012,0],
          "bottle":[0.03,0],
          "juice1":[0.05,2],
          "juice2":[0.055,3],
        }
    #resource list, fixed cost（未使用）, upper bound
    #corbon foot print (CFP) data (kg/km), variable term of corbon foot print (CFPV) (kg/ton km)
    ResourceData={
          "line1": [100,100,0, 0],
          "line2": [20,100,0, 0],
          "vehicle": [100,100,0.8, 0.1],
          "ship": [30,180, 0.2, 0.1]
        }

    #Resource-Prod data (resource usage)
    ResourceProdData = {
            ("line1", "juice1"):  1,
            ("line2", "juice1"):  1,
            ("line1", "juice2"):  1,
            ("line2", "juice2"):  1,
            ("vehicle", "juice1"):  1,
            ("ship", "juice1"):     1,
            ("vehicle", "juice2"):  1,
            ("ship", "juice2"):     1,
            ("vehicle", "apple"):   1,
            ("vehicle", "melon"):  1,
            ("ship", "apple"):     1,
            ("ship", "bottle"):     1,
    }

    #ArcResource list, fixed cost (ArcResourceFC)
    ArcResourceData={
            ("source1", "plantin","vehicle"):  10,
            ("source2", "plantin","ship"):  30,
            ("plantin", "plantout","line1"):  50,
            ("plantin", "plantout","line2"):  100,
            ("plantout", "customer","vehicle"):  20,
            ("plantout", "customer","ship"):  40,
        }

    #ArcResourceProd data,
    # type: 0=transport, 1=assemble, 2=dis-assemble,
    # variable cost, cycle time, lead time (LT), and upper bound of flow volume (UB)
    ArcResourceProdData=    {
            ("source1", "plantin","vehicle","apple"): [0,1,1,10,50],
            ("source1", "plantin","vehicle","melon"): [0,2,2,20,50],
            ("source2", "plantin","ship","apple"):    [0,3,3,15,50],
            ("source2", "plantin","ship","bottle"):   [0,3,3,15,50],

            ("plantin", "plantout","line1","juice1"): [1,10,5,1,10],
            ("plantin", "plantout","line1","juice2"): [1,10,5,1,10],
            ("plantin", "plantout","line2","juice1"): [1,5,3,1,10],
            ("plantin", "plantout","line2","juice2"): [1,5,3,1,10],

            ("plantout", "customer","vehicle","juice1"): [0,10,5,3,10],
            ("plantout", "customer","vehicle","juice2"): [0,10,5,4,10],
            ("plantout", "customer","ship","juice1"): [0,2,2,10,10],
            ("plantout", "customer","ship","juice2"): [0,2,2,12,10],
        }

    #Node-Prod data; value, demand (negative values are supply), DPENALTY+, DPENALTY- (demand violation penalties)
    NodeProdData ={
            ("source1", "apple"):  [10,-30,0,10000],
            ("source1", "melon"):  [10,-50,0,10000],
            ("source2", "apple"):  [5,-100,0,10000],
            ("source2", "bottle"): [5,-100,0,10000],

            ("plantin", "apple"): [15,0,0,0],
            ("plantin", "melon"): [15,0,0,0],
            ("plantin", "bottle"): [8,0,0,0],

            ("plantout", "juice1"): [150,0,0,0],
            ("plantout", "juice2"): [160,0,0,0],
            ("customer", "juice1"): [170,10,0,10000],
            ("customer", "juice2"): [180,10,0,10000],
    }

    #BOM
    Unit = {
            ("juice1", "apple"):  2,
            ("juice1", "melon"):  1,
            ("juice1", "bottle"):  1,
            ("juice2", "apple"):  2,
            ("juice2", "melon"):  2,
            ("juice2", "bottle"):  1
    }

    phi = {}
    for (q,p) in Unit:
        phi[p,q] = Unit[q,p] 

    ratio = 5.0  #interest ratio
    SSR = 1.65   #safety stock ratio
    CFPUB =400.0 #upper bound of carbon foot print (kg)

    ArcList, ArcFC, Distance= gp.multidict(ArcData)     
    Prod, Weight, VAR = gp.multidict(ProdData)
    Child, Parent ={}, {}
    for (p,q) in phi:
        if q in Child:
            Child[q].append(p)
        else:
            Child[q] = [p] 
        if p in Parent:
            Parent[p].append(q)
        else:
            Parent[p] = [q]

    Res, ResourceFC, ResourceUB, CFP, CFPV = gp.multidict(ResourceData)

    ResourceProd, R = gp.multidict(ResourceProdData)

    ArcResource, ArcResourceFC = gp.multidict(ArcResourceData)
    ArcResourcePair= gp.tuplelist([(i,j,r) for (i,j,r) in ArcResource])

    ArcResourceProd, Type, VariableCost, CycleTime, LT, UB = gp.multidict(ArcResourceProdData)
    ArcResourceProdPair= gp.tuplelist(ArcResourceProd)
    TransPair, AsmblPair, DisasmblPair =[],[],[]
    for (i,j,r,p) in Type:
        if Type[i,j,r,p]==1:
            AsmblPair.append( (i,j,r,p) )
        elif Type[i,j,r,p]==2:
            DisasmblPair.append( (i,j,r,p) )
        else:
            TransPair.append( (i,j,r,p) )

    NodeProd, VAL, Demand, DP_plus, DP_minus =gp.multidict(NodeProdData)
    NodeProdPair=gp.tuplelist(NodeProd)
    DemandNodeProdPair=[(i,p) for (i,p) in Demand if Demand[i,p]>0]
    SupplyNodeProdPair=[(i,p) for (i,p) in Demand if Demand[i,p]<0]

    products, arcs, nodes = {}, {}, {}
    activities, modes, resources = {}, {}, {}

    model = Model(name="Generic LND + Safety Stock Allocation")

    for (i,j) in ArcList:
        nodes[i] = Node(name=f"node({i})")
        nodes[j] = Node(name=f"node({j})")
        arcs[i,j] = model.addArc(name=f"arc({i},{j})", source= nodes[i], sink=nodes[j]) #need fixed_cost?

    for (i,j,r) in ArcResource:
        resources[i,j,r] = model.addResource(name=f"resource({i},{j},{r})", capacity=ResourceUB[r]) #need fixed_cost?

    for p in Prod:
        products[p] = model.addProduct(name=f"product({p})")

    for (i,j,r,p) in ArcResourceProd:
        if (i,j,p) not in activities:
            if Type[i,j,r,p]==0: #輸送
                activities[i,j,p] = model.addActivity(name=f"act({i},{j},{p})", atype="transport", product=products[p]) 
            else: #製造
                activities[i,j,p] = model.addActivity(name=f"act({i},{j},{p})", atype="make", product=products[p])
        mode = Mode(name=f"mode({r})", variable_cost=VariableCost[i,j,r,p], fixed_cost= ArcResourceFC[i,j,r])
        mode.addResource(resource = resources[i,j,r], variable= R[r,p]) 
        if Type[i,j,r,p] == 1: #assemble
            for q in Child[p]:
                mode.addComponent(component = products[q], quantity= phi[q,p])
        elif Type[i,j,r,p] == 2: #disassemble
            for q in Parent[p]:
                mode.addByproduct(byproduct = products[q], quantity= phi[p,q])
        activities[i,j,p].addMode(mode)
        arcs[i,j].addActivity(activity=activities[i,j,p])

    for(i,p) in Demand:
        if Demand[i,p]>0:
            model.addData(dtype="demand", product=products[p], node=nodes[i], amount=Demand[i,p]  )
        elif Demand[i,p]<0: #供給
            model.addData(dtype="supply", product=products[p], node=nodes[i], amount= -Demand[i,p], under_penalty =0.  )


    #多期間モデル（在庫も考慮）に拡張
    for i,t in enumerate(pd.date_range(start="2024/1/1", end="2024/2/28", freq="w")):
        model.addPeriod(name= f"period({i})", start=t)
    model.update()
    planning_horizon = len(model.period_list)
    print("T=",planning_horizon)

    #工場の入口にボトルの在庫を置く
    bottle = model.products["product(bottle)"]
    plantin = model.nodes["node(plantin)"]
    act_inv_bottle = model.addActivity(name="act bottle inventory", atype="inventory", product= bottle)
    #mode_inv_bottle = Mode(name="mode bottle inventory")  #在庫が資源を必要としたり，上限がある場合にはモードで定義
    #act_inv_bottle.addMode(mode_inv_bottle)
    plantin.addActivity(act_inv_bottle)
    model.addData(dtype="value",product=bottle, node=plantin, amount=10.) #製品の価値を入力
    model.addData(dtype="inventory",product=bottle, node=plantin, period= model.periods["period(0)"],amount=30.) #製品の在庫量を入力
    model.addData(dtype="inventory",product=bottle, node=plantin, period= model.periods["period(7)"],amount=10.)
    #計画期間を与えて， 過去の実績を定数として入力し，計画期間内だけ最適化

    #工場の出口にジュース1の在庫を置く
    juice1 = model.products["product(juice1)"]
    plantout = model.nodes["node(plantout)"]
    act_inv_juice1 = model.addActivity(name="act juice inventory", atype="inventory", product= juice1)
    plantout.addActivity(act_inv_juice1)
    model.addData(dtype="value",product=juice1, node=plantout, amount=40.) 

    model.interest_rate = 0.15 #在庫保管比率を設定
    #model.period_index

    #単一期間の場合にはstart=0, end=1を指定する．
    start_period_name = 'period(1)'
    end_period_name = 'period(7)'
    gp_model = optimize(model, start_period_name, end_period_name)

    return model, gp_model
```

```{.python.marimo}
# f = open('model4.json', 'w')
# f.write(model.model_dump_json(exclude_none=True))
# f.close()
# g, bom = visualize(model, size=50)
# g
```

```{.python.marimo}
generate_general_lndp_button = mo.ui.run_button(label="Generate and Optimize General Logistics Network Design Problem")
generate_general_lndp_button
```

```{.python.marimo}
mo.stop(
    not generate_general_lndp_button.value,
    mo.md(text="一般化ロジスティクス・ネットワーク設計問題を表示するには上のボタンを押してください。"),
)
_model, _gp_model = generate_general_lndp()
_g, _bom = visualize(_model, size=20)
mo.image(_g.pipe(format='png'))
```

## Example: Risk Exposure Index

サプライチェーンのリスクを評価するための指標 (Risk Exposure Index) を計算するためのモデル


集合：

-  $G=(V,E)$: 工場の輸送可能関係を表すグラフ；点の添え字を $f,g$ とする．
-  $BOM$: 部品展開表(BOM)を表すグラフ; ノード（製品を表す） $p$ の子ノードの集合を $CHILD_p$ とする．ノードの添え字を $p,q$ とする．
-  $D=(N,A)$: 製品の移動関係を表すグラフ；点の添え字を $i,j$ とする．点は工場 $f$ と製品 $p$ の組であり， $i=(f,p)$ の関係がある．
また、$(i,j)\in A$ であるのは、$i=(g,q), j=(f,p)$ としたとき、
$(i,j) \in E$（輸送可能）かつ $q \in CHILD_p$ （子製品・親製品の関係）を満たすときに限るものとする。
以下では、点、枝はこのグラフの点、枝を指すものとする。
-  $PRODUCT_f$: 工場 $f$ で生産可能な製品の集合

パラメータ：

- $\phi_{pq}$: 親製品$p \in P$を1ユニット製造するのに必要な子製品$q \in Child_p$の部品数
-  $R_{ij}$: 枝 $(i,j)$ での製品の変換比率；上の $\phi_{pq}$ をもとに $R_{ij} = R_{(g,q),(f,p)}=\phi_{pq}$ と計算される．
-  $I_i$: 点 $i (\in N)$ におけるパイプライン在庫量
-  $d_i$: 点 $i (\in N)$ における単位期間内での需要量
-  $UB_i$: 点 $i (\in N)$ における生産可能量上限（$0$ のとき途絶中であることを表す．）
-  $C_f$: 工場 $f$ の単位期間内での生産容量


変数：

-  $y_{ij}$: 点 $i$ から点 $j$ への輸送量
-  $u_i$: 点 $i (\in N)$ 上での生産量
-  $\tau$: 余裕生存時間（TTS:Time-to-Survival）

定式化

$$
\begin{array}{l l l}
     \max & \tau  &    \\
      s.t.      &  u_j \leq \sum_{i=(g,q)} \frac{1}{R_{ij}} y_{ij} & \forall j=(f,p), q \in CHILD_p \\
                 &  \sum_{j} y_{ij} \leq u_i + I_i   & \forall i \in N \\
                 &  d_i \tau \leq　u_i +I_i               & \forall i \in N \\
                 & \sum_{p \in PRODUCT_f} u_{fp} \leq C_f \tau     & \forall f \in V \\
                 & 0 \leq u_i \leq UB_i  & \forall i \in N \\
                 & y_{ij} \geq 0  & \forall (i,j) \in A
\end{array}
$$

最初の制約（生産量と入庫輸送量の関係）：
工場 $f$ における製品 $p$ の生産量 $u_{j} (j=(f,p))$ は、その部品 $q$ の輸送量以下でなくてはいけない。
ここで、輸送量は出発地点 $i$ における子製品 $q$ の単位で表現されているので、親製品の量は変換比率 $R_{ij}$ で割らなければならない。

2番目の制約（生産量と出庫輸送量の関係）：
点 $i$ から出る輸送量 $y_{ij}$ は、生産量 $u_i$ とパイプライン在庫量 $I_i$ の和以下でなくてはならない。

3番目の制約（需要満足条件）：
生産量は途絶期間内の需要量以上でなければならない。

4番目の制約（工場の生産容量制約）：
工場 $f$ で生産される総量は、その容量 $C_f$ 以下でなければならない。

各点（工場と製品の組）が途絶したシナリオ（点 $i$ の生産量上限 $UB_i$ を $0$ に設定） において上の問題を解き、そのときの目的関数値が、点の途絶時に品切れがおきない最大の期間（余裕生存時間）となる。

```{.python.marimo}
def generate_risk():
    BOM=nx.DiGraph() #BOM: bill of materials
    #  工場グラフ
    #  1 => 0
    #  2 => 0
    Capacity = {0: 300, 1: 500, 2: 200 }
    Products  = {0: ['P4','P5'], 1:['P1','P3'], 2: ['P2','P3']}
    BOM.add_weighted_edges_from([ ('P1','P4', 1), ('P2','P5',2),('P3','P5',1) ])

    model = Model(name = "Risk Exposure Index")
    products, activities, modes, nodes, arcs, resources = {}, {}, {}, {}, {}, {} 
    #製品
    for p in [f"P{i}" for i in range(1,6)]:
        products[p] = model.addProduct(name=p)
    #ノードとアーク
    for i in range(3):
        nodes[i] = model.addNode(name=i)
    for (i,j) in [(1,0),(2,0)]:
        arcs[i,j] = model.addArc(name=f"arc({i},{j})", source=nodes[i], sink=nodes[j])
    #点の上に資源と活動を配置
    for i, prods in Products.items():
        resources[i] = model.addResource(name=f"res({i})", capacity=Capacity[i] )
        for p in prods:
            activities[i,p] = model.addActivity(name=f"act({i},{p})", atype="inventory", product = products[p])
            modes[i,p] = Mode(name=f"mode({i},{p})", upper_bound = 100.)
            for q in BOM.predecessors(p):
                modes[i,p].addComponent(component=products[q], quantity=BOM[q][p]["weight"] )
            modes[i,p].addResource(resource=resources[i], variable=1.)
            activities[i,p].addMode(mode=modes[i,p])
            nodes[i].addActivity(activities[i,p])
    #パイプライン在庫，需要，製品の価値データ
    for i, prods in Products.items():
        for p in prods:
            model.addData(dtype="inventory", product=products[p], node=nodes[i], amount= 300.)
            if p in ["P4", "P5"]:
                model.addData(dtype="value", product=products[p], node=nodes[i], amount= 4.)
                model.addData(dtype="demand", product=products[p], node=nodes[i], amount= 100.)
            else:
                model.addData(dtype="value", product=products[p], node=nodes[i], amount= 1.)


    G = nx.DiGraph() #plant graph
    for (i,j) in model.arcs:
        G.add_edge(i,j)
    BOM = nx.DiGraph() #BOM graph
    ProductsInPlant = defaultdict(list)
    UB = {} 
    for i, node in model.nodes.items():
        for a, act in node.activities.items():
            ProductsInPlant[i].append(act.product.name)
            for m, mode in act.modes.items():
                UB[i,act.product.name] = mode.upper_bound
    for a, act in model.activities.items():
        p = act.product.name
        for m, mode in act.modes.items():
            if mode.components is not None:
                for q, weight in mode.components.items():
                    BOM.add_edge(q,p,weight=weight)

    ProdGraph = nx.tensor_product(G,BOM)

    Temp = ProdGraph.copy()
    for (i,p) in Temp:
        if p not in ProductsInPlant[i]:
            ProdGraph.remove_node( (i,p) )
    print("ProdGraph Nodes=",ProdGraph.nodes())
    print("ProdGraph Edges=",ProdGraph.edges())

    Pipeline, Demand = {}, {}
    for (dtype,p,i,t) in model.data:
        if dtype=="demand": 
            Demand[i,p] = model.data[dtype,p,i,t].amount
        elif dtype =="inventory":
            Pipeline[i,p] = model.data[dtype,p,i,t].amount
    R={}
    for (u,v) in ProdGraph.edges():
        (i,p)=u
        (j,q)=v
        R[u,v]=BOM[p][q]['weight']
    print("R=",R)
    print("Demand=",Demand)
    print("UB=", UB) 
    print("Capacity=", Capacity)

    #最適化モデル
    survival_time = []
    tempUB = {}

    for s in ProdGraph:
        # for each scenario s
        for n in ProdGraph:
            tempUB[n] = UB[n]
        tempUB[s] = 0.0
        #print("Scenario", s)

        gp_model = gp.Model()
        tn = gp_model.addVar(name='tn', vtype='C')
        u, y = {}, {}
        for i, j in ProdGraph.edges():
            y[i, j] = gp_model.addVar(name=f'y({i},{j})')
        for j in ProdGraph:
            u[j] = gp_model.addVar(name=f'u({j})', ub=tempUB[j])

        if GUROBI: gp_model.update()

        gp_model.setObjective(tn, gp.GRB.MAXIMIZE)

        # 生産量と入庫輸送量との関係
        for j in ProdGraph:
            if ProdGraph.in_degree(j) > 0:
                (plant, prod) = j
                for child in BOM.predecessors(prod):
                    gp_model.addConstr(u[j] <= gp.quicksum((1/float(R[i, j])) * y[i, j]
                                                     for i in ProdGraph.predecessors(j)
                                                     if i[1] == child),
                                    name=f"BOM{j}_{child}")

        # 生産量と出庫輸送量の関係
        for i in ProdGraph:
            if ProdGraph.out_degree(i) > 0:
                gp_model.addConstr(gp.quicksum(y[i, j] for j in ProdGraph.successors(i))
                                <= u[i] + Pipeline[i], name= f"BOM2_{i}")

        # 需要満足条件1
        for j in Demand:
            gp_model.addConstr(u[j]+Pipeline[j] >= Demand[j]*tn, name=f"Demand{j}")

        # # 需要満足条件2
        # for f in TotalDemand:
        #     gp_model.addConstr(gp.quicksum( u[f,p] for p in Product[f]) >= TotalDemand[f]*tn, name=f"TotalDemand{f}")

        # 工場の生産容量制約
        for f in Capacity:
            gp_model.addConstr(gp.quicksum(u[f, p] for p in ProductsInPlant[f]) <= Capacity[f]*tn,
                            name=f"Capacity{f}")

        gp_model.Params.OutputFlag = False
        gp_model.optimize()
        #print('tn=', tn.X)
        survival_time.append(int(tn.X*10)/10)
    print("生存時間=", survival_time)

    return model, gp_model
```

```{.python.marimo}
# g, bom = visualize(model, size=50)
# g
```

```{.python.marimo}
# bom
```

### Optimization

```{.python.marimo}
generate_risk_button = mo.ui.run_button(label="Generate and Optimize Risk Optimization Problem")
generate_risk_button
```

```{.python.marimo}
mo.stop(
    not generate_risk_button.value,
    mo.md(text="リスク最適化問題を表示するには上のボタンを押してください。"),
)
_model, _gp_model = generate_risk()
_g, _bom = visualize(_model, size=20)
mo.image(_g.pipe(format='png'))
```

## Example: Safety Stock Allocation

点集合 $N$ は在庫地点を表し，枝 $(i,j) \in A$ が存在するとき，在庫地点 $i$ は在庫地点 $j$ に補充
を行うことを表す．複数の在庫地点から補充を受ける点においては，補充された各々の品目を用いて
別の品目を生産すると考える．

点 $i$ における品目の生産時間は，$T_i$ 日である．
$T_i$ には各段階における生産時間の他に，待ち時間および輸送時間も含めて考える．

このとき点 $j$ は，複数の在庫地点から補充を受けるので，点 $j$ が品目を発注してから，すべての品目が揃うまで生産を開始できない．

点上で生産される品目は，点によって唯一に定まるものと仮定する．
このとき，点と品目を表す添え字は同一であると考えられるので，
有向グラフ $G=(N,A)$ は部品展開表を意味することになる．
$(i,j) \in A$ のとき，点 $j$ 上の品目 $j$ は，点 $i$ から補充される品目 $i$ をもとに生産される．
品目 $j$ を生産するのに必要な品目 $i$ の量を $\phi_{ij}$ と記す．

需要は，後続する点をもたない点 $j$ 上で発生するものとし，
その $1$ 日あたりの需要量は，期待値 $\mu_j$ の定常な分布をもつものとする．
点 $j$ における $t$ 日間における需要の最大値を $D_j(t)$ 記す．

直接の需要をもたない点（後続点をもつ点） $i$ に対する需要量の期待値 $\mu_i$ は，

$$
 \mu_i = \sum_{(i,j) \in A} \phi_{ij} \mu_j
$$

と計算できる．

点 $i$ に対する$t$ 日間における需要の最大値 $D_i(t)$ は，

$$
  D_i(t)= t \mu_i +\left( \sum_{(i,j) \in A} \phi_{ij} (D_j(t)-t \mu_j)^p \right)^{1/p}
$$

と計算されるものと仮定する．

ここで，$p  (\geq 1)$ は定数である．
$p$ が大きいほど，需要の逆相関が強いことを表す．
点 $j$ 上での需要が常に同期していると仮定した場合には $p=1$ が，
点 $j$ 上での需要が独立な正規分布にしたがうと仮定した場合には $p=2$ が推奨される．


点 $i$ の保証リード時間を $L_i$ と記す．
ここでは，各点 $i$ に対して，保証リード時間の下限は $0$ とし，上限は $\bar{L}_i$ であるとする．

点 $j$ が品目を発注してから，すべての品目が
揃うまでの時間（日数）を入庫リード時間とよび， $LI_j$ と記す．

点 $j$ における入庫リード時間 $LI_j$ は，以下の式を満たす．

$$
  L_i \leq LI_j \ \ \  \forall (i,j) \in A
$$

入庫リード時間 $LI_i$ に生産時間 $T_i$ を加えたものが，補充の指示を行ってから生産を完了するまでの時間となる．
これを，補充リード時間とよぶ．
補充リード時間から、保証リード時間 $L_i$ を減じた時間内の最大需要に相当する在庫を保持していれば，在庫切れの心配がないことになる．
補充リード時間から $L_i$ を減じた時間（$L_{i+1}+T_i-L_i$）を正味補充時間とよぶ．

点 $i$ における安全在庫量 $I_i$ は，正味補充時間内における最大需要量から平均需要量を減じた量であるので，

$$
  I_i= D(LI_{i}+T_i-L_i) -  (LI_{i}+T_i-L_i) \mu_i
$$

となる．

記述を簡単にするために，点 $i$ において，保証リード時間が $L$，
入庫リード時間が $LI$ のときの安全在庫費用を表す，以下の関数 $HC_i(L,LI)$ を導入しておく．

$$
 HC_i (L,LI)=h_i \left\{ D(LI+T_i-L) - (LI+T_i-L) \mu_i \right\}
$$

上の記号を用いると，木ネットワークモデルにおける安全在庫配置問題は，以下のように定式化できる．

$$
\begin{array}{l l l }
  minimize & \sum_{i \in N} HC_i (L_i,LI_i) &            \\
  s.t.
                & L_i \leq LI_{i}+T_i  & \forall i \in N  \\
                & L_i \leq LI_j       &  \forall (i,j) \in A  \\
                & 0 \leq L_i \leq \bar{L}_i      & \forall i \in N
\end{array}
$$

ここで，最初の制約式は，正味補充時間が非負であることを表す．

```{.python.marimo}
def tabu_search_for_SSA(
    G, ProcTime, z, mu, sigma, h, LTUB, max_iter=100, TLLB=1, TLUB=10, seed=1
):
    """
    一般のネットワークの安全在庫配置モデルに対するタブー探索（mu未使用；NRT日の最大在庫量をシミュレーションもしくは畳み込みによって事前計算？もしくは正規分布で近似）
    """
    assert nx.is_directed_acyclic_graph(G)
    np.random.seed(seed)
    n = len(G)
    b = np.random.randint(0, 2, n)
    candidate = []
    for i in G:
        candidate.append(i)
    m = len(candidate)
    NRT = np.zeros(n)
    MaxLI = np.zeros(n)
    MinLT = np.zeros(n)
    vNRT = np.zeros((m, n))
    vMaxLI = np.zeros((m, n))
    vMinLT = np.zeros((m, n))
    TabuList = np.zeros(m, int)
    for i in G.down_order():
        if G.in_degree(i) == 0:
            MaxLI[i] = ProcTime[i]
        else:
            max_ = 0.0
            for k in G.predecessors(i):
                max_ = max(max_, (1 - b[k]) * MaxLI[k])
            MaxLI[i] = ProcTime[i] + max_
    for i in G.up_order():
        if G.out_degree(i) == 0:
            MinLT[i] = LTUB[i]
        else:
            min_ = np.inf
            for j in G.successors(i):
                min_ = min(min_, NRT[j] + MinLT[j] - ProcTime[j])
            MinLT[i] = min_
        NRT[i] = max(MaxLI[i] - MinLT[i], 0)
    cost = (h * z * sigma * np.sqrt(NRT)).sum()
    best_cost = cost
    prev_cost = cost
    best_sol = b.copy()
    b_prev = b.copy()
    best_NRT = NRT.copy()
    best_MaxLI = MaxLI.copy()
    best_MinLT = MinLT.copy()
    ltm_factor = 0.0
    ltm_increase = cost / float(n * max_iter) / 10.0
    ltm = np.zeros(m, int)
    for iter_ in range(max_iter):
        B = []
        for i in candidate:
            newb = b.copy()
            newb[i] = 1 - b[i]
            B.append(newb)
        B = np.array(B)
        for i in G.down_order():
            if G.in_degree(i) == 0:
                vMaxLI[:, i] = ProcTime[i]
            else:
                max_ = np.zeros(m)
                for k in G.predecessors(i):
                    max_ = np.maximum(max_, (1 - B[:, k]) * vMaxLI[:, k])
                vMaxLI[:, i] = ProcTime[i] + max_
        for i in G.up_order():
            if G.out_degree(i) == 0:
                vMinLT[:, i] = LTUB[i]
            else:
                min_ = np.full(m, np.inf)
                for j in G.successors(i):
                    min_ = np.minimum(
                        min_, vNRT[:, j] + vMinLT[:, j] - ProcTime[j]
                    )
                vMinLT[:, i] = min_
            vNRT[:, i] = np.maximum(vMaxLI[:, i] - vMinLT[:, i], 0)
        cost = (h * z * sigma * np.sqrt(vNRT[:, :])).sum(axis=1)
        min_ = np.inf
        istar = -1
        for i in range(m):
            if iter_ >= TabuList[i]:
                if cost[i] + ltm_factor * ltm[i] < min_:
                    min_ = cost[i] + ltm_factor * ltm[i]
                    istar = i
            elif cost[i] < best_cost:
                if cost[i] < min_:
                    min_ = cost[i]
                    istar = i
        if istar == -1:
            TLLB = max(TLLB - 1, 1)
            TLUB = max(TLUB - 1, 2)
            TabuList = np.zeros(m, int)
        else:
            b = B[istar]
            ltm[istar] = ltm[istar] + 1
            if np.all(b_prev == b):
                TLLB = TLLB + 1
                TLUB = TLUB + 1
            elif prev_cost == cost[istar]:
                ltm_factor = ltm_factor + ltm_increase
            b_prev = b.copy()
            prev_cost = cost[istar]
            TabuList[istar] = iter_ + np.random.randint(TLLB, TLUB + 1)
            if cost[istar] < best_cost:
                best_cost = cost[istar]
                best_sol = B[istar].copy()
                best_NRT = vNRT[istar].copy()
                best_MaxLI = vMaxLI[istar].copy()
                best_MinLT = vMinLT[istar].copy()
    for i in range(n):
        if best_NRT[i] <= 1e-05:
            best_sol[i] = 0
    return (best_cost, best_sol, best_NRT, best_MaxLI, best_MinLT)
```

```{.python.marimo}
def generate_safety_stock():
    n = 7
    G = SCMGraph()
    for i in range(n):
        G.add_node(i)
    G.add_edges_from([(0, 2), (1, 2), (2,4), (3,4), (4,5), (4,6)])
    #点のラベルの付け替え
    mapping ={i:idx for idx,i in enumerate(G)}
    G = nx.relabel_nodes(G, mapping=mapping, copy=True)

    z = np.full(len(G),1.65)
    h = np.array([1,1,3,1,5,6,6])
    mu = np.array([200,200,200,200,200,100,100])
    sigma = np.array([14.1,14.1,14.1,14.1,14.1,10,10])
    ProcTime = np.array([6,2,3,3,3,3,3], int) 
    LTUB = np.array([0,0,0,0,0,4,3],int) #最大保証リード時間（需要地点のみ意味がある）

    # max_iter = 10
    # seed =123
    # TLLB, TLUB = 2,10  # tabu length is a random number between (TLLB, TLUB)
    # best_cost, best_sol, best_NRT, best_MaxLI, best_MinLT  = tabu_search_for_SSA(G, ProcTime, z, mu, sigma, 
    #                                                                    h, LTUB, max_iter = 10, TLLB =1, TLUB =3, seed = seed)

    # print("最良値", best_cost)
    # print("最良解", best_sol)
    # print("正味補充時間", best_NRT)
    # print("最大補充リード時間", best_MaxLI)
    # print("最小保証リード時間", best_MinLT)

    model = Model(name="Safety Stock Allocation")
    products, activities, modes ={},{},{}

    for i in G:
        products[i] = model.addProduct(name=f"prod({i})", value=h[i])
        activities[i] = model.addActivity(name=f"act({i})", atype="inventory", product=products[i] )
        modes[i] = Mode(name=f"mode({i})", duration=ProcTime[i], service_time = LTUB[i], service_level = 0.95) 
        activities[i].addMode(modes[i])

    for i in G:
        for k in G.predecessors(i):
            modes[i].addComponent(component=products[k], quantity=1)

    for i,p in enumerate(products): 
        model.addData(dtype="demand", product=products[p], amount= mu[i], std = sigma[i] )

    G = SCMGraph()
    value, demand, std, duration, service_time, service_level = {},{},{},{},{},{}
    for a, act in model.activities.items():
        p = act.product.name
        value[p] = act.product.value
        for m, mode in act.modes.items():
            duration[p] = mode.duration if mode.duration is not None else 0
            service_time[p] = mode.service_time if mode.service_time is not None else 0
            service_level[p] = mode.service_level
            if mode.components is not None: 
                for q in mode.components: 
                    G.add_edge(q,p)
    for (dtype,p,_,_), data in model.data.items():
        if data.dtype=="demand":
            demand[p] = data.amount
            std[p] = data.std

    mapping ={i:idx for idx,i in enumerate(G)}
    G = nx.relabel_nodes(G, mapping=mapping, copy=True)

    h = np.zeros( len(G), float)
    mu = np.zeros( len(G), float)
    sigma = np.zeros( len(G), float)
    ProcTime = np.zeros( len(G), float)
    z = np.zeros( len(G), float)
    for p in model.products:
        idx = mapping[p]
        h[idx] = value[p]
        mu[idx] = demand[p]
        sigma[idx] = std[p]
        ProcTime[idx] = duration[p]
        LTUB[idx] = service_time[p]
        z[idx] = scipy.stats.norm.ppf( service_level[p] ) #他の分布でも対応可能

    max_iter = 10
    TLLB, TLUB = 2,10  # tabu length is a random number between (TLLB, TLUB)
    seed = 1
    best_cost, best_sol, best_NRT, best_MaxLI, best_MinLT  = tabu_search_for_SSA(G, ProcTime, z, mu, sigma, 
                                                                       h, LTUB, max_iter = max_iter, TLLB =TLLB, TLUB =TLUB, seed = seed)
    print("最良値", best_cost)
    print("最良解", best_sol)
    print("正味補充時間", best_NRT)
    print("最大補充リード時間", best_MaxLI)
    print("最小保証リード時間", best_MinLT)

    return model
```

```{.python.marimo}
# g, bom = visualize(model)
# bom
```

### Optimization

```{.python.marimo}
generate_ssa_button = mo.ui.run_button(label="Generate and Optimize Safety Stock Allocation Problem")
generate_ssa_button
```

```{.python.marimo}
mo.stop(
    not generate_ssa_button.value,
    mo.md(text="安全在庫配置問題を表示するには上のボタンを押してください。"),
)
_model = generate_safety_stock()
_g, _bom = visualize(_model, size=20)
mo.image(_g.pipe(format='png'))
mo.image(_bom.pipe(format='png'))
```

## Example: Unit Commitment

**起動停止問題**(unit commitment problem)とは，発電機の起動と停止を最適化するためのモデルであり，
各日の電力需要を満たすように時間ごとの発電量を決定する．

定式化のためには，動的ロットサイズ決定問題に以下の条件を付加する必要がある．

* 一度火を入れると $\alpha$ 時間は停止できない．
* 一度停止すると $\beta$ 時間は再稼働できない．

この制約の定式化が実用化のための鍵になるので，簡単に説明しておく．

**集合:**

* $P$: 発電機（ユニット）の集合

**変数:**

* $y_t^p$: 期 $t$ に発電機 $p$ が稼働するとき $1$

* $z_t^p$: 期 $t$ に発電機 $p$ が稼働を開始(switch-on)するとき $1$

* $w_t^p$:  期 $t$ に発電機 $p$ が停止を開始(switch-off)するとき $1$


上の変数の関係は以下の式で表すことができる．

$$
\begin{array}{l l }
  z_t^p \leq y_t^p               &  \forall p \in P, t=1,2, \ldots,T \\
  z_t^p -w_{t}^p = y_t^p -y_{t-1}^p &  \forall p \in P, t=1,2, \ldots,T
\end{array}
$$


- 開始したら最低でも $\alpha$ 期 は連続稼働：

$\Rightarrow$ 「期 $t$ に稼働していない」　 **ならば**　 「$t-\alpha+1$ から $t$ までは開始できない」

$\Rightarrow$ $y_t^p$ ならば $z_{s}^p=0$ ($\forall s=t-\alpha+1,\ldots,t$)

$$
\displaystyle\sum_{s=t-\alpha+1}^t z_{s}^p \leq y_t^p  \ \ \  \forall t=\alpha,\alpha+1, \ldots,T
$$

弱い定式化：

$\Rightarrow$ 「期 $t$ に開始した」 **ならば** 「$t$ から$t+\alpha+1$ までは稼働」

$$
\alpha z_{t}^p \leq \displaystyle\sum_{s=t}^{t+\alpha-1} y_t^p \ \ \  \forall t=1,2, \ldots,T+1-\alpha
$$

- 稼働を終了したら，再び稼働を開始するためには，$\beta$ 期以上：

$\Rightarrow$ 「期 $t$ に稼働した」　 **ならば** 「$t-\beta+1$ から $t$ までは終了できない」

$\Rightarrow$ $y_t^p=1$ ならば $w_{s}^p=0$ ($\forall s=t-\beta+1,\ldots,t$)

$$
\displaystyle\sum_{s=t-\beta+1}^{t} w_{s}^p \leq 1-y_t^p  \ \ \  \forall t=\beta,\beta+1, \ldots,T
$$

弱い定式化：

$\Rightarrow$ 「期 $t$ に停止した」　**ならば**  「$t$から $t+\beta-1$ までは稼働しない」

$$
\beta w_{t}^p \leq \displaystyle\sum_{s=t}^{t+\beta-1} (1-y_t^p) \ \ \  \forall t=1,2, \ldots,T+1-\beta
$$

```{.python.marimo}
import marimo as mo
```