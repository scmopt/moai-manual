---
title: Optseq
marimo-version: 0.10.12
width: full
---

# スケジューリング最適化システム OptSeq

>  Scheduling Solver OptSeq
<!---->
<p><a target="_blank" href="https://colab.research.google.com/github/scmopt/moai-manual/blob/main/optseq-trial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
 <a class="reference external" href="https://studiolab.sagemaker.aws/import/github/scmopt/moai-manual/blob/main/optseq-trial.ipynb"><img alt="Open In SageMaker Studio Lab" src="https://studiolab.sagemaker.aws/studiolab.svg" /></a></p>

```{.python.marimo}
# | export
import sys
import pathlib
import os
import re
import copy
import platform
import string
import datetime as dt
import ast
import pickle
from collections import Counter, defaultdict

trans = str.maketrans(":-+*/'(){}^=<>$ |#?,\¥", "_" * 22)  # 文字列変換用
# 以下非標準ファイル
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly

# import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots

from intervaltree import Interval, IntervalTree #gantt chart描画の高速化で用いる

# Pydantic
from typing import List, Optional, Union, Tuple, Dict, Set, Any, DefaultDict, ClassVar
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    validator,
    confloat,
    conint,
    constr,
    Json,
    PositiveInt,
    NonNegativeInt,
)
from pydantic.tools import parse_obj_as
from datetime import datetime, date, time

import graphviz

from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.chart import ScatterChart, Reference, Series
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.formatting.rule import ColorScaleRule, CellIsRule, FormulaRule
from openpyxl.styles import Color, PatternFill, Font, Border, Alignment
from openpyxl.styles.borders import Border, Side
from openpyxl.comments import Comment

import networkx as nx
```

## はじめに

**スケジューリング**（scheduling）とは，稀少資源を諸活動へ（時間軸を考慮して）割り振るための方法に対する理論体系である．
スケジューリングの応用は，工場内での生産計画，計算機におけるジョブのコントロール，プロジェクトの遂行手順の決定など，様々である．

ここで考えるのは，以下の一般化資源制約付きスケジューリングモデルであり，ほとんどの実際問題をモデル化できるように設計されている．

- 複数の作業モードをもつ作業
- 時刻依存の資源使用可能量上限
- 作業ごとの納期と重み付き納期遅れ和
- 作業の後詰め
- 作業間に定義される一般化された時間制約
- モードごとに定義された時刻依存の資源使用量
- モードの並列処理
- モードの分割処理
- 状態の考慮

OptSeq（オプトシーク）は，一般化スケジューリング問題に対する最適化ソルバーである．
スケジューリング問題は，通常の混合整数最適化ソルバーが苦手とするタイプの問題であり，
実務における複雑な条件が付加されたスケジューリング問題に対しては，専用の解法が必要となる．
OptSeqは，スケジューリング問題に特化した**メタヒューリスティクス**(metaheuristics)を用いることによって，
大規模な問題に対しても短時間で良好な解を探索することができるように設計されている．


OptSeqは， http://logopt.com/optseq/ からダウンロードするか，以下のように pipコマンドでインストールできる．

```python
pip instal optseq
```

また，例題や練習問題は，[アナリティクス練習問題集のサポートページ](https://scmopt.github.io/analytics/14optseq.html) にある．
<!---->
### OptSeqの基本クラス

行うべき仕事（ジョブ，作業，タスク）を**作業**(activity；活動)とよぶ． スケジューリング問題の目的は作業をどのようにして時間軸上に並べて遂行するかを決めることであるが，
ここで対象とする問題では作業を処理するための方法が何通りかあって，そのうち1つを選択することによって
処理するものとする．このような作業の処理方法を**モード**(mode)とよぶ．

納期や納期遅れのペナルティ（重み）は作業ごとに定めるが，
作業時間や資源の使用量はモードごとに決めることができる．

作業を遂行するためには**資源**(resource)を必要とする場合がある．
資源の使用可能量は時刻ごとに変化しても良いものとする．
また，モードごとに定める資源の使用量も作業開始からの経過時間によって変化しても良いものとする．
通常，資源は作業完了後には再び使用可能になるものと仮定するが，
お金や原材料のように一度使用するとなくなってしまうものも考えられる．
そのような資源を**再生不能資源**(nonrenewable resource)とよぶ．

作業間に定義される**時間制約**(time constraint)は，
ある作業（先行作業）の処理が終了するまで，別の作業（後続作業）の処理が開始できないことを表す
先行制約を一般化したものであり，
先行作業の開始（完了）時刻と後続作業の開始（完了）時刻の間に以下の制約があることを規定する．

> 先行作業の開始（完了）時刻 $+$ 時間ずれ $\leq$ 後続作業の開始（完了）時刻

ここで，時間ずれは任意の整数値であり負の値も許すものとする． この制約によって，作業の同時開始，最早開始時刻，時間枠などの様々な条件を記述することができる．

OptSeqでは，モードを作業時間分の小作業の列と考え，処理の途中中断や並列実行も可能であるとする．その際，中断中の資源使用量や並列作業中の資源使用量も別途定義できるものとする．

また，時刻によって変化させることができる**状態**(state)が準備され，モード開始の状態の制限やモードによる状態の推移を定義できる．

<img src="https://github.com/scmopt/scmopt_data/blob/main/fig/optseqclass.jpg?raw=true" width=600 height=200>
<!---->
### 注意

OptSeqでは作業，モード，資源名を文字列で区別するため重複した名前を付けることはできない．
なお，使用できる文字列は, 英文字 (a--z, A--Z), 数字 (0--9), 大括弧 ([ ]),  アンダーバー (_), および @ に限定される．
また，作業名は source, sink以外， モードは dummy 以外の文字に限定される．

それ以外の文字列はすべてアンダーバー (_)に置き換えられる．
<!---->
## Parameters クラス

OptSeqを制御するためのパラメータを格納したクラス．

Modelクラスの中で使用される．

OptSeqに内在されている最適化ソルバーの動作は， **パラメータ**(parameter)を変更することによってコントロールできる．
モデルインスタンスmodelのパラメータを変更するときは，以下の書式で行う．


> model.Params.パラメータ名 = 値


代表的なパラメータとその意味を記す．

- TimeLimitは最大計算時間 (秒) を設定する． 既定値は600 (秒)
- RandomSeedは乱数系列の種を設定する．既定値は 1
- Makespanは最大完了時刻（一番遅く終わる作業の完了時刻）を最小にするときTrueそれ以外のとき （各作業に定義された納期遅れの重みと再生不能資源の逸脱量の重み付き和を最小にするとき）False（既定値）を設定する．
- Neighborhoodは近傍探索を行う際の近傍数を設定するためのパラメータである． 既定値は20であり，大規模な問題例で求解時間が長い場合には小さい値に設定することが推奨される．
- Tenure: タブー長の初期値を表すパラメータ．必ず正数を入力する．既定値は1．
- Initial: 前回最適化の探索を行った際の最良解を初期値とした探索を行うときTrue，それ以外のときFalseを表すパラメータである． 既定値はFalse
最良解の情報は作業の順序と選択されたモードとしてファイル名 optseq_best_act_data.txtに保管されている． このファイルを書き換えることによって，異なる初期解から探索を行うことも可能である．
- OutputFlag: 計算の途中結果を出力させるためのフラグである． 2のとき詳細出力， 1(True)のとき出力On， 0(False)のとき出力Off． 既定値は0(False)で何も表示しない．
- ReportInterval: 計算のログを出力するためのインターバル．既定値は1073741823
- Backtruck: 最大のバックトラック数を表すパラメータ．既定値は1000

```{.python.marimo}
# | export
class Parameters(BaseModel):
    """
    OptSeq parameter class to control the operation of OptSeq.

    - param  TimeLimit: Limits the total time expended (in seconds). Positive integer. Default=600.

    - param  OutputFlag: Controls the output log. Integer. Default=1.

    - param  RandomSeed: Sets the random seed number. Integer. Default=1.

    - param  ReportInterval: Controls the frequency at which log lines are printed (iteration number). Default=1073741823.

    - param  Backtruck: Controls the maximum backtrucks. Default=1000.

    - param  MaxIteration: Sets the maximum numbers of iterations. Default=1073741823.

    - param  Initial: =True if the user wants to set an initial activity list. Default = False.

            Note that the file name of the activity list must be "optseq_best_act_data.txt."

    - param  Tenure: Controls a parameter of tabu search (initial tabu tenure). Default=1.
    - param  Neighborhood: Controls a parameter of tabu search (neighborhood size). Default=20.
    - param  Makespan: Sets the objective function.

            Makespan is True if the objective is to minimize the makespan (maximum completion time),
            is False otherwise, i.e., to minimize the total weighted tardiness of activities.
            Default=False.
    """

    TimeLimit: PositiveInt = 600
    OutputFlag: int = 1  # ON
    RandomSeed: int = 1
    ReportInterval: PositiveInt = 1073741823
    Backtruck: PositiveInt = 1000
    MaxIteration: PositiveInt = 1073741823
    Initial: bool = False
    Tenure: PositiveInt = 1
    Neighborhood: PositiveInt = 20
    Makespan: bool = False

    def __str__(self) -> str:
        return f""" 
 TimeLimit = {self.TimeLimit} \n OutputFlag = {self.OutputFlag}
 RandomSeed = {self.RandomSeed} \n ReportInterval = {self.ReportInterval} \n Backtruck = {self.Backtruck}
 Initial = {self.Initial} \n Tenure = {self.Tenure}
 Neighborhood = {self.Neighborhood} \n makespan = {self.Makespan} """
```

### Parametesクラスの使用例

```{.python.marimo}
_params = Parameters()
_params.Makespan = False
print(_params)
```

## State クラス

Stateは状態を定義するためのクラスである．

状態インスタンスは，モデルに含まれる形で生成される．
状態インスタンスは，モデルの状態追加メソッド(addState)の返値として生成される．

> 状態インスタンス = model.addState(name）


状態インスタンスは，指定時に状態の値を変化させるためのメソッドaddValueをもつ．

addValue(time, value)は，状態を時刻time（非負整数値）に値value（非負整数値）に変化させることを指定する．


状態インスタンスは以下の属性をもつ．

- nameは状態の名称を表す文字列である．
- valueは，時刻をキーとし，その時刻に変化する値を値とした辞書である．


状態によってモードの開始が制限されている場合には， 作業のautoselect属性をTrueに設定しておくことが推奨される．
ただし，作業の定義でautoselectをTrueに指定した場合には，その作業に制約を逸脱したときの重みを無限大とした （すなわち絶対制約とした）再生不能資源を定義することはできない．
かならず重みを既定値の無限大'inf'ではない正数値と設定し直す必要がある．

```{.python.marimo}
class State(BaseModel):
    """
    OptSeq state class.

    You can create a state object by adding a state to a model (using Model.addState)
    instead of by using a State constructor.

        - Arguments:
            - name: Name of state. Remark that strings in OptSeq are restricted to a-z, A-Z, 0-9,[],_ and @.

    """
    ID: ClassVar[int] = 0
    name: Optional[str] = ''
    Value: Optional[Dict[NonNegativeInt, NonNegativeInt]] = {}

    def __init__(self, name: str='') -> None:
        super().__init__(name=name)
        if name == '' or name == None:
            name = '__s{0}'.format(State.ID)
            State.ID = State.ID + 1
        if type(name) != str:
            raise ValueError('State name must be a string')
        self.name = str(name).translate(trans)

    def __str__(self) -> str:
        ret = ['state {0} '.format(self.name)]
        for v in self.Value:
            ret.append('time {0} value {1} '.format(v, self.Value[v]))
        return ' '.join(ret)

    def addValue(self, time: NonNegativeInt=0, value: NonNegativeInt=0) -> None:
        """
        Adds a value to the state
            - Arguments:
                - time: the time at which the state changes.
                - value: the value that the state changbes to

            - Example usage:

            >>> state.addValue(time=5,value=1)
        """
        if type(time) == type(1) and type(value) == type(1):
            self.Value[time] = value
        else:
            print('time and value of the state {0} must be integer'.format(self.name))
            raise TypeError
```

### Stateクラスの使用例

```{.python.marimo}
_state = State(name="sample state")
_state.addValue(time=5, value=1)
print(_state)
```

## Mode クラス

OptSeqでは， 作業の処理方法を**モード**(mode)とよぶ．
Modeは作業(Activity)の遂行方法を定義するためのクラスである．
作業は少なくとも1つのモードをもち，そのうちのいずれかを選択して処理される．

モードのインスタンスは，モードクラスModeから生成される．

> モードインスタンス = Mode(name, duration=1)

コンストラクタの引数の名前と意味は以下の通り．

- nameはモードの名前を文字列で与える．ただしモードの名前に'dummy'を用いることはできない．
- durationはモードの作業時間を非負の整数で与える．既定値は $1$．

モードインスタンスは，以下のメソッドをもつ．

- addResourceはモードを実行するときに必要な資源とその量を指定する．

  > モードインスタンス.addResource(resource, requirement={}, rtype = None)

  引数と意味は以下の通り．


  - resourceは追加する資源インスタンスを与える．
  - requirementは資源の必要量を辞書もしくは正数値で与える．
   辞書のキーはタプル (開始時刻,終了時刻) であり， 値は資源の使用量を表す正数値である．
   正数値で与えた場合には，開始時刻は $0$， 終了時刻は無限大と設定される．

   **注：**作業時間が $0$ のモードに資源を追加することはできない．その場合には実行不可能解と判断される．

  - rtypeは資源のタイプを表す文字列． None, 'break', 'max' のいずれかから選択する（既定値は通常の資源を表すNone）．
   'break'を与えた場合には，中断中に使用する資源量を指定する．
    'max'を与えた場合には，並列処理中に使用する資源の「最大量」を指定する．
   省略可で，その場合には通常の資源使用量を表し，並列処理中には資源使用量の「総和」を使用することになる．

- addBreakは中断追加メソッドである．
  モードは単位時間ごとに分解された作業時間分の小作業の列と考えられる．
  小作業を途中で中断してしばらく時間をおいてから次の小作業を開始することを**中断**(break)とよぶ．
  中断追加メソッド(addBreak)は，モード実行時における中断の情報を指定する．

  > モードインスタンス.addBreak(start=0, finish=0, maxtime='inf')

  引数と意味は以下の通り．


  - startは中断可能な最早時刻を与える．省略可で，既定値は $0$．
  - finishは中断可能時刻の最遅時刻を与える．省略可で，既定値は $0$．
  - maxtimeは最大中断可能時間を与える．省略可で，既定値は無限大（'inf')．


- addParallelは並列追加メソッドである．
  モードは単位時間ごとに分解された作業時間分の小作業の列と考えられる．
  資源量に余裕があるなら，同じ時刻に複数の小作業を実行することを**並列実行**(parallel execution)とよぶ．

  並列追加メソッドaddParallelは，モード実行時における並列実行に関する情報を指定する．

  > モードインスタンス.addParallel(start=1, finish=1, maxparallel='inf')

  引数と意味は以下の通り．


  - startは並列実行可能な最小の小作業番号を与える．省略可で，既定値は $1$．
  - finishは並列実行可能な最大の小作業番号を与える．省略可で，既定値は $1$．
  - maxparallelは同時に並列実行可能な最大数を与える．省略可で，既定値は無限大('inf')．


- addStateは状態追加メソッドである．
  状態追加メソッド(addState)は，モード実行時における状態の値と実行直後（実行開始が時刻 $t$ のときには，時刻 $t+1$）の
  状態の値を定義する．


   > モードインスタンス.addState(state, fromValue=0, toValue=0)


   引数と意味は以下の通り．


   - stateはモードに付随する状態インスタンスを与える．省略不可．
   - fromValueはモード実行時における状態の値を与える．省略可で，既定値は $0$．
   - toValueはモード実行直後における状態の値を与える．省略可で，既定値は $0$．

モードインスタンスは以下の属性をもつ．

- nameはモード名である．
- durationはモードの作業時間である．
- requirementはモードの実行の資源・資源タイプと必要量を表す辞書である．
キーは資源名と資源タイプ（None:通常， ’break’:中断中， ’max’:並列作業中の最大資源量）のタプルであり，
値は資源必要量を表す辞書である．この辞書のキーはタプル（開始時刻，終了時刻）であり，値は資源の使用量を表す正数値である．

- breakableは中断の情報を表す辞書である．
辞書のキーはタプル（開始時刻，終了時刻）であり，値は中断可能時間を表す正数値である．

- parallelは並列作業の情報を表す辞書である．辞書のキーはタプル(開始小作業番号,終了小作業番号)であり，
値は最大並列可能数を表す正数値である．

```{.python.marimo}
class Mode(BaseModel):
    """
    OptSeq mode class.

        - Arguments:
            - name: Name of mode (sub-activity).
                    Remark that strings in OptSeq are restricted to a-z, A-Z, 0-9,[],_ and @.
                    Also you cannot use "dummy" for the name of a mode.
                    - duration(optional): Processing time of mode. Default=0.

        - Attbibutes:
            - requirement: Dictionary that maps a pair of resource name and resource type (rtype) to requirement dictionary.
                    Requirement dictionary maps intervals (pairs of start time and finish time) to amounts of requirement.
                    Resource type (rtype) is None (standard resource type), "break" or "max."
            - breakable: Dictionary that maps breakable intervals to maximum brek times.
            - paralel:  Dictionary that maps parallelable intervals to maximum parallel numbers.
            - state: Dictionary that maps states to the tuples of values.
    """
    ID: ClassVar[int] = 0
    name: Optional[str] = ''
    duration: Optional[PositiveInt] = 1
    requirement: Optional[Dict[Tuple[str, Optional[str]], Dict[Tuple[NonNegativeInt, NonNegativeInt], NonNegativeInt]]] = None
    breakable: Optional[Dict[Tuple[NonNegativeInt, NonNegativeInt], PositiveInt]] = None
    parallel: Optional[Dict[Tuple[PositiveInt, PositiveInt], PositiveInt]] = None
    state: Optional[Dict[State, Tuple[int, int]]] = None

    def __init__(self, name: str='', duration: PositiveInt=1, requirement: Optional[Dict[Tuple[str, Optional[str]], Dict[Tuple[NonNegativeInt, NonNegativeInt], NonNegativeInt]]]=None, breakable: Optional[Dict[Tuple[NonNegativeInt, NonNegativeInt], PositiveInt]]=None, parallel: Optional[Dict[Tuple[PositiveInt, PositiveInt], PositiveInt]]=None, state: Optional[Dict[State, Tuple[int, int]]]=None):
        super().__init__(name=name, duration=duration, requirement=requirement, breakable=breakable, parallel=parallel, state=state)
        if name is None or name == '':
            name = '__m{0}'.format(Mode.ID)
            Mode.ID = Mode.ID + 1
        if name == 'dummy':
            print("'dummy' cannnot be used as a mode name")
            raise NameError("'dummy' cannnot be used as a mode name")
        if type(name) != str:
            raise ValueError('Mode name must be a string')
        self.name = str(name).translate(trans)

    def __str__(self) -> str:
        ret = [' duration {0} '.format(self.duration)]
        if self.requirement:
            for r, rtype in self.requirement:
                for interval, cap in self.requirement[r, rtype].items():
                    s, t = interval
                    if rtype == 'max':
                        ret.append(' {0} max interval {1} {2} requirement {3} '.format(r, s, t, cap))
                    elif rtype == 'break':
                        ret.append(' {0} interval break {1} {2} requirement {3} '.format(r, s, t, cap))
                    elif rtype == None:
                        ret.append(' {0} interval {1} {2} requirement {3} '.format(r, s, t, cap))
                    else:
                        print('resource type error')
                        raise TypeError('resource type error')
        if self.breakable:
            for interval, cap in self.breakable.items():
                s, t = interval
                if cap == 'inf':
                    ret.append(' break interval {0} {1} '.format(s, t))
                else:
                    ret.append(' break interval {0} {1} max {2} '.format(s, t, cap))
        if self.parallel:
            for interval, cap in self.parallel.items():
                s, t = interval
                if cap == 'inf':
                    ret.append(' parallel interval {0} {1} '.format(s, t))
                else:
                    ret.append(' parallel interval {0} {1} max {2} '.format(s, t, cap))
        if self.state:
            for s in self.state:
                for f, t in self.state[s]:
                    ret.append(' {0} from {1} to {2} '.format(s, f, t))
        return ' \n'.join(ret)

    def addState(self, state: 'State', fromValue: int=0, toValue: int=0) -> None:
        """
        Adds a state change information to the mode.

            - Arguments:
                - state: State object to be added to the mode.
                - fromValue: the value from which the state changes by the mode
                - toValue:  the value to which the state changes by the mode

            - Example usage:

            >>> mode.addState(state1,0,1)

            defines that state1 is changed from 0 to 1.

        """
        if self.state is None:
            self.state = {}
        if type(fromValue) != type(1) or type(toValue) != type(1):
            print('time and value of the state {0} must be integer'.format(self.name))
            raise TypeError('time and value of the state {0} must be integer'.format(self.name))
        elif state.name not in self.state:
            self.state[state.name] = [(fromValue, toValue)]
        else:
            self.state[state.name].append((fromValue, toValue))

    def addResource(self, resource: 'Resource', requirement: Union[int, Dict[Tuple[NonNegativeInt, Union[NonNegativeInt, str]], NonNegativeInt]]=None, rtype: Optional[str]=None):
        """
        Adds a resource to the mode.

            - Arguments:
                - resurce: Resource object to be added to the mode.
                - requirement: Dictionary that maps intervals (pairs of start time and finish time) to amounts of requirement.
                               It may be an integer; in this case, requirement is converted into the dictionary {(0,"inf"):requirement}.
                - rtype (optional): Type of resource to be added to the mode.
                None (standard resource type; default), "break" or "max."

            - Example usage:

            >>> mode.addResource(worker,{(0,10):1})

            defines worker resource that uses 1 unit for 10 periods.

            >>> mode.addResource(machine,{(0,"inf"):1},"break")

            defines machine resource that uses 1 unit during break periods.

            >>> mode.addResource(machine,{(0,"inf"):1},"max")

            defines machine resource that uses 1 unit during parallel execution.
        """
        if self.requirement is None:
            self.requirement = {}
        if type(requirement) == type(1):
            requirement = {(0, 'inf'): requirement}
        if type(resource.name) != type('') or type(requirement) != type({}):
            print(f"type error in adding a resource {resource.name} to activity's mode {self.name}: requirement type is {type(requirement)}")
            raise TypeError('type error in adding a resource {0} to activity {1}'.format(resource.name, self.name))
        elif rtype == None or rtype == 'break' or rtype == 'max':
            if (resource.name, rtype) not in self.requirement:
                self.requirement[resource.name, rtype] = {}
            data = copy.deepcopy(self.requirement[resource.name, rtype])
            data.update(requirement)
            self.requirement[resource.name, rtype] = data
        else:
            print('rtype must be None or break or max')
            raise NameError('rtype must be None or break or max')

    def addBreak(self, start: NonNegativeInt=0, finish: NonNegativeInt=0, maxtime: Union[NonNegativeInt, str]='inf') -> None:
        """
        Sets breakable information to the mode.

            - Arguments:
                - start(optional): Earliest break time. Non-negative integer. Default=0.
                - finish(optional): Latest break time.  Non-negative integer or "inf." Default=0.
                    Interval (start,finish) defines a possible break interval.
                - maxtime(optional): Maximum break time. Non-negative integer or "inf." Default="inf."

            - Example usage:

            >>> mode.addBreak(0,10,1)

            defines a break between (0,10) for one period.
        """
        if self.breakable is None:
            self.breakable = {}
        data = copy.deepcopy(self.breakable)
        data.update({(start, finish): maxtime})
        self.breakable = data

    def addParallel(self, start: PositiveInt=1, finish: PositiveInt=1, maxparallel: Union[NonNegativeInt, str]='inf'):
        """
        Sets parallel information to the mode.

            - Arguments:
                - start(optional): Smallest job index executable in parallel. Positive integer. Default=1.
                - finish(optional): Largest job index executable in parallel. Positive integer or "inf." Default=1.
                - maxparallel(optional): Maximum job numbers executable in parallel. Non-negative integer or "inf." Default="inf."

            - Example usage:

            >>> mode.addParallel(1,1,2)
        """
        if self.parallel is None:
            self.parallel = {}
        data = copy.deepcopy(self.parallel)
        data.update({(start, finish): maxparallel})
        self.parallel = data
```

### Modeクラスの使用例

```{.python.marimo}
_mode = Mode(name="test mode", duration=10)
_mode.addBreak(0, 10, 1)
_mode.addParallel(1, 1, 2)
print(_mode)
```

## Activity クラス

成すべき仕事（ジョブ，活動，タスク）を総称して**作業**(activity)とよぶ．

Acticityは作業を定義するためのクラスである．

作業クラスのインスタンスは，モデルインスタンスmodelの作業追加メソッド(addActivity)の返値として生成される．

> 作業インスタンス=model.addActivity(name="", duedate="inf", backward= False, weight=1, autoselect=False, quadratic=False)

作業には任意の数のモード（作業の実行方法）を追加することができる．
モードの追加は，addModesメソッドで行う．

> 作業インスタンス.addModes(モードインスタンス1, モードインスタンス2, ... )

作業の情報は，作業インスタンスの属性に保管されている．
作業インスタンスは以下の属性をもつ．

- nameは作業名である．
- duedateは作業の納期であり，$0$ 以上の整数もしく無限大'inf'（既定値）を入力する．
- backwardは作業を後ろ詰め（バックワード）で最適化するときTrue，それ以外の場合（前詰め；フォワード；既定値）Falseを入力する．
ただし，後ろ詰めを指定した場合には，状態変数は機能しない．また，納期 (duedate) は無限大 'inf'以外であるか、後続作業に 'inf' 以外の納期が設定されている必要がある．
また，前詰めと後ろ詰めの混合も可能であるが，後ろ詰めを指定した作業の後続作業も「自動的に」後ろ詰めに変更される．
後ろ詰めの場合の納期は**絶対条件**として処理されるので，後続作業の含めて実行不能にならないように設定する必要がある．

- weightは作業の完了時刻が納期を遅れたときの単位時間あたりのペナルティである．

- autoselectはモードを自動選択するときTrue，それ以外のときFalseを設定する．
   既定値はFalseであり，状態によってモードの開始が制限されている場合には， autoselectをTrueに設定しておくことが推奨される．

**注意:**
作業の定義でautoselectをTrueに指定した場合には，その作業に制約を逸脱したときの重みを無限大とした
（すなわち絶対制約とした）再生不能資源を定義することはできない．
かならず重みを既定値の無限大'inf'ではない正数値と設定し直す必要がある．

- quadraticは納期遅れに対する関数を線形ではなく，2次関数にしたいときTrueにする．既定値はFalse．

- modesは作業に付随するモードインスタンスのリストを保持する．
- selectedは探索によって発見された解において選択されたモードインスタンスを保持する．
- startは探索によって発見された解における作業の開始時刻である．
- completionは探索によって発見された解における作業の終了時刻である．
- executeは探索によって発見された解における作業の実行を表す辞書である．キーは作業の開始時刻と終了時刻のタプル，
値は並列実行数を表す正数値である．

```{.python.marimo}
class Activity(BaseModel):
    """
    OptSeq activity class.

        You can create an activity object by adding an activity to a model (using Model.addActivity)
        instead of by using an Activity constructor.

        - Arguments:
                - name: Name of activity. Remark that strings in OptSeq are restricted to a-z, A-Z, 0-9,[],_ and @.
                        Also you cannot use "source" and "sink" for the name of an activity.
                - duedate(optional): Duedate of activity. A non-nagative integer or string "inf."
                - backward(optional): True if activity is distached backwardly, False (default) otherwise.
                - weight(optional): Panalty of one unit of tardiness. Positive integer.
                - autoselect(optional): True or False flag that indicates the activity selects the mode automatically or not.
    """
    ID: ClassVar[int] = 0
    name: Optional[str] = ''
    duedate: Optional[Union[NonNegativeInt, str]] = 'inf'
    backward: Optional[bool] = False
    weight: Optional[PositiveInt] = 1
    autoselect: Optional[bool] = False
    quadratic: Optional[bool] = False
    modes: Optional[List[Mode]] = []
    start: Optional[NonNegativeInt] = 0
    completion: Optional[NonNegativeInt] = 0
    execute: Optional[Dict[Tuple[NonNegativeInt, NonNegativeInt], NonNegativeInt]] = {}
    selected: Optional[Mode] = None

    def __init__(self, name='', duedate='inf', backward=False, weight=1, autoselect=False, quadratic=False):
        super().__init__(name=name, duedate=duedate, backward=backward, weight=weight, autoselect=autoselect, quadratic=quadratic)
        if name == 'source' or name == 'sink':
            print(" 'source' and 'sink' cannnot be used as an activity name")
            raise NameError
        if type(name) != str:
            raise ValueError('Activity name must be a string')
        if name == '' or name == None:
            name = '__a{0}'.format(Activity.ID)
            Activity.ID = Activity.ID + 1
        self.name = str(name).translate(trans)

    def __str__(self) -> str:
        ret = ['activity {0}'.format(self.name)]
        if self.duedate != 'inf':
            if self.backward == True:
                ret.append(' backward duedate {0} '.format(self.duedate))
            else:
                ret.append(' duedate {0} '.format(self.duedate))
            ret.append(' weight {0} '.format(self.weight))
            if self.quadratic:
                ret.append(' quad ')
        if self.autoselect == True:
            ret.append(' autoselect ')
        for m in self.modes:
            ret.append(' {0} '.format(m.name))
        return ' \n'.join(ret)

    def addModes(self, *modes: List[Mode]) -> None:
        """
        Adds a mode or modes to the activity.

            - Arguments:
                - modes: One or more mode objects.

            - Example usage:

            >>> activity.addModes(mode1,mode2)
        """
        for mode in modes:
            self.modes.append(mode)
```

### Activityクラスの使用例

```{.python.marimo}
_act = Activity("sample activity", duedate=100, backward=True, weight=10)
_mode = Mode(name="test mode", duration=10)
_act.addModes(_mode)
print(_act)
```

## Resource クラス

Resourceは資源を定義するためのクラスである．

資源インスタンスは，モデルの資源追加メソッド(addResource)の返値として生成される．

> 資源インスタンス = model.addResource(name, capacity, rhs=0, direction='<=', weight='inf'）

資源インスタンスは，以下のメソッドをもつ．


- addCapacityは資源に容量を追加するためのメソッドであり，資源の容量を追加する．

   引数と意味は以下の通り．


   - startは資源容量追加の開始時刻（区間の始まり）を与える．
   - finishは資源容量追加の終了時刻（区間の終わり）を与える．
   - amountは追加する容量（資源量上限）を与える．

- setRhs(rhs)は再生不能資源を表す線形制約の右辺定数をrhsに設定する．引数は整数値（負の値も許すことに注意）とする．

- setDirection(dir)は再生不能資源を表す制約の種類をdirに設定する． 引数のdirは'<=', '>=', '='のいずれかとする．

- addTerms(coeffs,vars,values)は，再生不能資源制約の左辺に1つ，もしくは複数の項を追加するメソッドである． 作業がモードで実行されるときに $1$， それ以外のとき $0$ となる変数（値変数）を x[作業,モード]とすると，  追加される項は，

$$
 係数 \times x[作業,モード]
$$

と記述される． addTermsメソッドの引数は以下の通り．

  - coeffsは追加する項の係数もしくは係数リスト．係数もしくは係数リストの要素は整数（負の値も許す）．
  - varsは追加する項の作業インスタンスもしくは作業インスタンスのリスト． リストの場合には，リストcoeffsと同じ長さをもつ必要がある．
  - valuesは追加する項のモードインスタンスもしくはモードインスタンスのリスト． リストの場合には，リストcoeffsと同じ長さをもつ必要がある．


資源インスタンスは以下の属性をもつ．

- nameは資源名である．
- capacityは資源の容量（使用可能量の上限）を表す辞書である．
   辞書のキーはタプル (開始時刻, 終了時刻) であり，値は容量を表す正数値である．
- rhsは再生不能資源制約の右辺定数である． 既定値は $0$．
- directionは再生不能資源制約の方向を表す．　既定値は '<='．
- termsは再生不能資源制約の左辺を表す項のリストである．各項は (係数,作業インスタンス,モードインスタンス) のタプルである．
- weightは再生不能資源制約を逸脱したときのペナルティの重みを表す． 正数値か絶対制約を表す'inf'を入れる． 既定値は無限大（絶対制約）を表す文字列'inf'である．
- residualは最適化後に計算される資源の余裕量を表す辞書である． 辞書のキーはタプル (開始時刻, 終了時刻) であり，値は残差を表す非負整数値である．

```{.python.marimo}
class Resource(BaseModel):
    """
    OptSeq resource class.

         - Arguments:
             - name: Name of resource.
                     Remark that strings in OptSeq are restricted to a-z, A-Z, 0-9,[],_ and @.
             - capacity (optional): Capacity dictionary of the renewable (standard) resource.
                         Capacity dictionary maps intervals (pairs of start time and finish time) to amounts of capacity.
                         If it is given by a positive integer, it is converted into the dictionay {(0,"inf"):capacity}.
             - rhs (optional): Right-hand-side constant of nonrenewable resource constraint.
             - direction (optional): Rirection (or sense) of nonrenewable resource constraint; "<=" (default) or ">=".
             - weight (optional): Weight of nonrenewable resource to compute the penalty for violating the constraint.
                                  Non-negative integer or "inf" (default).

         - Attbibutes:
             - capacity: Capacity dictionary of the renewable (standard) resource.
             - rhs: Right-hand-side constant of nonrenewable resource constraint.
             - direction: Rirection (or sense) of nonrenewable resource constraint; "<=" (default) or "=" or ">=".
             - terms: List of terms in left-hand-side of nonrenewable resource.
                        Each term is a tuple of coeffcient, activity and mode.
             - weight: Weight of nonrenewable resource to compute the penalty for violating the constraint.
                       Non-negative integer or "inf" (default).
             - residual: Residual dictionary of the renewable (standard) resource.

    """
    ID: ClassVar[int] = 0
    name: Optional[str] = ''
    capacity: Optional[Union[int, Dict[Tuple[NonNegativeInt, NonNegativeInt], PositiveInt]]] = None
    rhs: Optional[int] = 0
    direction: Optional[str] = '<='
    weight: Optional[Union[NonNegativeInt, str]] = 'inf'
    terms: Optional[List[Tuple[int, Activity, Mode]]] = []
    residual: Optional[Dict[Tuple[NonNegativeInt, NonNegativeInt], NonNegativeInt]] = {}

    def __init__(self, name='', capacity: Optional[Union[int, Dict[Tuple[NonNegativeInt, NonNegativeInt], PositiveInt]]]=None, rhs: int=0, direction: str='<=', weight: Union[NonNegativeInt, str]='inf'):
        super().__init__(capacity=capacity, rhs=rhs, direction=direction, weight=weight)
        if capacity is None:
            capacity = {}
        if name is None or name == '':
            name = '__r{0}'.format(Resource.ID)
            Resource.ID = Resource.ID + 1
        if type(name) != str:
            raise ValueError('Resource name must be a string')
        self.name = str(name).translate(trans)
        if type(capacity) == type(1):
            self.capacity = {(0, 'inf'): capacity}

    def __str__(self) -> str:
        ret = []
        if self.capacity:
            ret.append('resource {0} '.format(self.name))
            capList = []
            for interval, cap in self.capacity.items():
                s, t = interval
                capList.append((s, t, cap))
            for s, t, cap in capList:
                ret.append(' interval {0} {1} capacity {2} '.format(s, t, cap))
        return ' \n'.join(ret)

    def addCapacity(self, start: NonNegativeInt=0, finish: NonNegativeInt=0, amount: PositiveInt=1) -> None:
        """
        Adds a capacity to the resource.

            - Arguments:
                - start(optional): Start time. Non-negative integer. Default=0.
                - finish(optional): Finish time. Non-negative integer. Default=0.
                 Interval (start,finish) defines the interval during which the capacity is added.
                - amount(optional): The amount to be added to the capacity. Positive integer. Default=1.

            - Example usage:

            >>> manpower.addCapacity(0,5,2)
        """
        data = copy.deepcopy(self.capacity)
        data.update({(start, finish): amount})
        self.capacity = data

    def printConstraint(self) -> str:
        """
        Returns the information of the linear constraint.

        The constraint is expanded and is shown in a readable format.
        """
        f = ['nonrenewable weight {0} '.format(self.weight)]
        if self.direction == '>=' or self.direction == '>':
            for coeff, var, value in self.terms:
                f.append('{0}({1},{2}) '.format(-coeff, var.name, value.name))
            f.append('<={0} \n'.format(-self.rhs))
        elif self.direction == '==' or self.direction == '=':
            for coeff, var, value in self.terms:
                f.append('{0}({1},{2}) '.format(coeff, var.name, value.name))
            f.append('<={0} \n'.format(self.rhs))
            f.append('nonrenewable weight {0} '.format(self.weight))
            for coeff, var, value in self.terms:
                f.append('{0}({1},{2}) '.format(-coeff, var.name, value.name))
            f.append('<={0} \n'.format(-self.rhs))
        else:
            for coeff, var, value in self.terms:
                f.append('{0}({1},{2}) '.format(coeff, var.name, value.name))
            f.append('<={0} \n'.format(self.rhs))
        return ''.join(f)

    def addTerms(self, coeffs: Union[int, List[int]], vars: Union[Activity, List[Activity]], values: Union[Mode, List[Mode]]) -> None:
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
        if type(coeffs) != type([]):
            self.terms.append((coeffs, vars, values))
        elif type(coeffs) != type([]) or type(vars) != type([]) or type(values) != type([]):
            print('coeffs, vars, values must be lists')
            raise TypeError('coeffs, vars, values must be lists')
        elif len(coeffs) != len(vars) or len(coeffs) != len(values) or len(values) != len(vars):
            print('length of coeffs, vars, values must be identical')
            raise TypeError('length of coeffs, vars, values must be identical')
        else:
            for i in range(len(coeffs)):
                self.terms.append((coeffs[i], vars[i], values[i]))

    def setRhs(self, rhs: int=0) -> None:
        """
        Sets the right-hand-side of linear constraint.

            - Argument:
                - rhs: Right-hand-side of linear constraint.

            - Example usage:

            >>> L.setRhs(10)

        """
        self.rhs = rhs

    def setDirection(self, direction: str='<=') -> None:
        if direction in ['<=', '>=', '=']:
            self.direction = direction
        else:
            print("direction setting error; direction should be one of '<=' or '>=' or '='")
            raise NameError("direction setting error; direction should be one of '<=' or '>=' or '='")
```

### Resourceクラスの使用例

```{.python.marimo}
_res = Resource(name='sample resource', capacity={(0, 10): 1})
_res.addCapacity(0, 5, 10)
print('renewable resource= \n', _res)
print()
_mode = Mode(name='test mode', duration=10)
_act = Activity('sample activity', duedate=100, backward=True, weight=10)
_act.addModes(_mode)
_res2 = Resource('non-renewable', rhs=1, direction='<=', weight=100)
_res2.addTerms(coeffs=1, vars=_act, values=_mode)
print(_res2.printConstraint())
```

## Temporal クラス

Temporalは時間制約を定義するためのクラスである．

時間制約インスタンスは，モデルに含まれる形で生成される．
時間制約インスタンスは，上述したモデルの時間制約追加メソッド(addTemporal)の返値として生成される．

> 時間制約インスタンス = model.addTemporal(pred, succ, tempType='CS', delay=0, pred_mode=None, succ_mode=None）

時間制約インスタンスは以下の属性をもつ．

- predは先行作業のインスタンスである．
- succは後続作業のインスタンスである．
- typeは時間制約のタイプを表す文字列であり，'SS'（開始，開始）,'SC'（開始，完了）,'CS'（完了，開始）,'CC'（完了，完了）のいずれかを指定する． 既定値は 'CS'
- delayは時間制約の時間ずれを表す整数値である． 既定値は $0$
- pred_modeは先行作業の特定のモードであり，そのモードに対して時間制約を追加したいときに使用する．既定値はNoneで，その場合にはモードに依存せず作業に対する時間制約となる．
- succ_modeは後続作業の特定のモードであり，そのモードに対して時間制約を追加したいときに使用する．既定値はNoneで，その場合にはモードに依存せず作業に対する時間制約となる．

```{.python.marimo}
# | export
class Temporal(BaseModel):
    """
    OptSeq temporal class.

    A temporal constraint has the following form::

        predecessor's completion (start) time +delay <=
                        successor's start (completion) time.

    Parameter "delay" can be negative.

        - Arguments:
            - pred: Predecessor (an activity object) or string "source."
                    Here, "source" specifies a dummy activity that precedes all other activities and starts at time 0.
            - succ: Successor (an activity object) or string "source."
                    Here, "source" specifies a dummy activity that precedes all other activities and starts at time 0.
            - tempType (optional): String that differentiates the temporal type.
                "CS" (default)=Completion-Start, "SS"=Start-Start,
                "SC"= Start-Completion, "CC"=Completion-Completion.
            - delay (optional): Time lag between the completion (start) times of two activities.
            - pred_mode (optional): Predecessor's mode
            - succ_mode (optional): Successor's mode

        - Attributes:
            - pred: Predecessor (an activity object) or string "source."
            - succ: Successor (an activity object) or string "source."
            - type: String that differentiates the temporal type.
                "CS" (default)=Completion-Start, "SS"=Start-Start,
                "SC"= Start-Completion, "CC"=Completion-Completion.
            - delay: Time lag between the completion (start) times of two activities. default=0.

    """

    pred: Union[Activity, str]  # activity or "source" or "sink"
    succ: Union[Activity, str]
    type: Optional[str] = "CS"
    delay: Optional[int] = 0
    pred_mode: Optional[Mode] = None
    succ_mode: Optional[Mode] = None

    def __init__(
        self,
        pred: Union[Activity, str],
        succ: Union[Activity, str],
        tempType: str = "CS",
        delay: int = 0,
        pred_mode: Optional[Mode] = None,
        succ_mode: Optional[Mode] = None,
    ) -> None:
        super().__init__(
            pred=pred,
            succ=succ,
            tempType=tempType,
            delay=delay,
            pred_mode=pred_mode,
            succ_mode=succ_mode,
        )

        if pred_mode is not None and pred_mode not in pred.modes:
            raise ValueError(f"Mode {pred_mode.name} is not in activity {pred.name}")
        if succ_mode is not None and succ_mode not in succ.modes:
            raise ValueError(f"Mode {succ_mode.name} is not in activity {succ.name}")

    def __str__(self) -> str:
        if self.pred == "source":
            pred = "source"
        elif self.pred == "sink":
            pred = "sink"
        else:
            pred = str(self.pred.name)

        if self.succ == "source":
            succ = "source"
        elif self.succ == "sink":
            succ = "sink"
        else:
            succ = str(self.succ.name)

        if self.pred_mode is None and self.succ_mode is None:
            # モードに依存しない時間制約
            ret = ["temporal {0} {1}".format(pred, succ)]
            ret.append(" type {0} delay {1} ".format(self.type, self.delay))
        else:
            # source,sink以外の場合で， 片方だけモード指定して，複数モードがある場合にはエラー
            if self.pred != "source" and self.succ != "sink":
                if self.pred_mode is None and self.succ_mode is not None:
                    raise ValueError(
                        f"The mode of activity {self.pred.name} is not specified!"
                    )
                if self.pred_mode is not None and self.succ_mode is None:
                    raise ValueError(
                        f"The mode of activity {self.succ.name} is not specified!"
                    )

            if self.pred == "source" or self.pred == "sink":
                pred_mode = "dummy"
            else:
                pred_mode = self.pred_mode.name

            if self.succ == "source" or self.succ == "sink":
                succ_mode = "dummy"
            else:
                succ_mode = self.succ_mode.name

            ret = [f"temporal {pred} mode {pred_mode} {succ} mode {succ_mode}"]
            ret.append(" type {0} delay {1} ".format(self.type, self.delay))

        # print(self.pred_mode, self.succ_mode, ret)
        return " ".join(ret)
```

### Temporalクラスの使用例

```{.python.marimo}
_act = Activity(name='sample activity')
_mode = Mode(name="test mode", duration=10)
_act.addModes(_mode)
_act2 = Activity(name='sample activity2')
_temp = Temporal(pred=_act, succ=_act2, delay=10)
_temp2 = Temporal(pred=_act, succ='sink', pred_mode=_act.modes[0])
print(_temp)
print(_temp2)
```

## Modelクラス

Modelはモデルを定義するためのクラスである．

Modelは引数なしで（もしくは名前を引数として），以下のように記述する．

> model = Model()

> model = Model('名前')


モデルインスタンスmodelは，以下のメソッドをもつ．

- addActivityは，モデルに1つの作業を追加する．返値は作業インスタンスである．


  > 作業インスタンス = model.addActivity(name="", duedate="inf", backward = False, weight=1, autoselect=False, quadratic =False)

  引数の名前と意味は以下の通り．

   - nameは作業の名前を文字列で与える．ただし作業の名前に'source', 'sink'を用いることはできない．

   - duedateは作業の納期を 0 以上の整数もしくは，無限大を表す文字列'inf'で与える． この引数は省略可で，既定値は'inf'である．

   - backwardは作業を後ろ詰め（バックワード）で最適化するときTrue，それ以外の場合（前詰め；フォワード；既定値）Falseを入力する．
   ただし，後ろ詰めを指定した場合には，**状態変数は機能しない**．また，納期 (duedate) は無限大 'inf'以外であるか、後続作業に 'inf' 以外の納期が設定されている必要がある．
   また，前詰めと後ろ詰めの混合も可能であるが，後ろ詰めを指定した作業の後続作業も「自動的に」後ろ詰めに変更される．後ろ詰めの場合の納期は**絶対条件**として処理されるので，後続作業の含めて実行不能にならないように設定する必要がある．

   - weightは作業の完了時刻が納期を遅れたときの単位時間あたりのペナルティである． 省略可で，既定値は 1．

   - autoselectは作業に含まれるモードを自動選択するか否かを表すフラグである． モードを自動選択するときTrue，それ以外のときFalseを設定する．
   既定値はFalse． 状態によってモードの開始が制限されている場合には， autoselectをTrueに設定しておくことが望ましい．

    **注意:**
    作業の定義でautoselectをTrueに指定した場合には，その作業に制約を逸脱したときの重みを無限大とした
    （すなわち絶対制約とした）再生不能資源を定義することはできない．
    かならず重みを既定値の無限大'inf'ではない正数値と設定し直す必要がある．

   - quadraticは納期遅れに対する関数を線形ではなく，2次関数にしたいときTrueにする．既定値はFalse．

- addResourceはモデルに資源を1つ追加する．返値は資源インスタンスである．


  > 資源インスタンス = model.addResource(name, capacity, rhs=0, direction='<=', weight='inf')


  引数の名前と意味は以下の通り．


   - nameは資源の名前を文字列で与える．
   - capacityは資源の容量（使用可能量の上限）を辞書もしくは正数値で与える．
   正数値で与えた場合には，開始時刻は $0$，終了時刻は無限大と設定される．
   辞書のキーはタプル (開始時刻, 終了時刻) であり，値は容量を表す正数値である．
   開始時刻と終了時刻の組を**区間**(interval)とよぶ． 離散的な時間を考えた場合には，時刻 $t-1$ から時刻 $t$ の区間を**期**(period) $t$ と定義する． 時刻の初期値を $0$ と仮定すると，期は $1$ から始まる整数値をとる． 区間 (開始時刻, 終了時刻) に対応する期は， 「開始時刻$+1$，開始時刻 $+2$， ...， 終了時刻」 となる．

   - rhsは再生不能資源制約の右辺定数を与える．省略可で，既定値は $0$．
   - directionは再生不能資源制約の種類（制約が等式か不等式か，不等式の場合には方向）を示す文字列を与える． 文字列は'<=', '>=', '=' のいずれかとする． 省略可であり，既定値は '<='である．
   - weightは 再生不能資源制約を逸脱したときのペナルティ計算用の重みを与える． 正数値もしくは無限大を表す文字列'inf'を入力する．省略可で，既定値は'inf'．

- addTemporalはモデルに時間制約を1つ追加する． 返値は時間制約インスタンスである．


  > 時間制約インスタンス = model.addTemporal(pred, succ, tempType='CS', delay=0, pred_mode=None, succ_mode=None)

  時間制約は，先行作業と後続作業の開始（もしくは完了）時刻間の関係を表し，
  以下のように記述される．

$$
先行作業（もしくはモード）の開始（完了）時刻 + 時間ずれ \leq 後続作業（もしくはモード）の開始（完了）時刻
$$

  ここで**時間ずれ**(delay)は時間の差を表す整数値である． 先行（後続）作業の開始時刻か完了時刻のいずれを対象とするかは，時間制約のタイプで指定する．
  タイプは，**開始時刻**(start time)のとき文字列'S'， **完了時刻**(completion time)のとき文字列'C'で表し，
  先行作業と後続作業のタイプを2つつなげて 'SS', 'SC', 'CS', 'CC'のいずれかから選択する．

  引数の名前と意味は以下の通り．

   - predは**先行作業**(predecessor)のインスタンスもしくは文字列'source'を与える． 文字列'source'は，すべての作業に先行する開始時刻 $0$ のダミー作業を定義するときに用いる．
   - succは**後続作業**(successor)のインスタンスもしくは文字列'sink'を与える． 文字列'sink'は，すべての作業に後続するダミー作業を定義するときに用いる．
   - tempTypeは時間制約のタイプを与える．
     'SS', 'SC', 'CS', 'CC'のいずれかから選択し，省略した場合の既定値は'CS'
   （先行作業の完了時刻と後続作業の開始時刻）である．
   - delayは先行作業と後続作業の間の時間ずれであり，整数値（負の値も許すことに注意）で与える． 既定値は $0$ である．

   - pred_modeは先行作業の特定のモードであり，そのモードに対して時間制約を追加したいときに使用する．既定値はNoneで，その場合にはモードに依存せず作業に対する時間制約となる．
   - succ_modeは後続作業の特定のモードであり，そのモードに対して時間制約を追加したいときに使用する．既定値はNoneで，その場合にはモードに依存せず作業に対する時間制約となる．



- addStateはモデルに状態を追加する．引数は状態の名称を表す文字列nameであり， 返値は状態インスタンスである．

  > 状態インスタンス = model.addState(name)

- optimizeはモデルの最適化を行う．返値はなし． 最適化を行った結果は，作業，モード，資源，時間制約インスタンスの属性に保管される． 引数は以下の通り．

  - cloud: 複数人が同時実行する可能性があるときTrue（既定値はFalse）; Trueのとき，ソルバー呼び出し時に生成されるファイルにタイムスタンプを追加し，計算終了後にファイルを消去する．
  - init_fn: 初期解の作業順とモードを設定するためのファイル名； 既定値は optseq_best_act_data.txt； パラメータのInitialがTrueのときのみ有効になる．
  - best_fn: 探索中に得られた最良解の情報（作業順とモード）を保管するためのファイル名； 既定値は optseq_best_act_data.txt


- writeは最適化されたスケジュールを簡易**Ganttチャート**(Gantt chart；Henry Ganttによって $100$年くらい前に提案されたスケジューリングの表記図式なので，Ganttの図式という名前がついている． 実は，最初の発案者はポーランド人のKarol Adamieckiで1896年まで遡る．） としてテキストファイルに出力する． 引数はファイル名(filename)であり，その既定値はoptseq.txtである．ここで出力されるGanttチャートは，作業別に選択されたモードや開始・終了時刻を示したものであり， 資源に対しては使用量と容量が示される．

- writeExcelは最適化されたスケジュールを簡易Ganttチャートとしてカンマ区切りのテキスト(csv)ファイルに出力する．引数はファイル名(filename)とスケールを表す正整数(scale)である．ファイル名の既定値はoptseq.csvである．スケールは，時間軸をscale分の $1$ に縮めて出力するためのパラメータであり，Excelの列数が上限値をもつために導入された．その既定値は $1$ である．なお，Excel用のGanttチャートでは，資源の残り容量のみを表示する．


モデルインスタンスは，モデルの情報を文字列として返すことができる．
たとえば，モデルインスタンスmodelの情報は，

>   print(model)

で得ることができる．作業，モード，資源，時間制約，状態のインスタンスについても同様であり，
print関数で情報を出力することができる．

モデルの情報は，インスタンスの属性に保管されている．インスタンスの属性は「インスタンス.属性名」でアクセスできる．

- actはモデルに含まれる作業インスタンスのリスト．
- resはモデルに含まれる資源インスタンスのリスト．
- activitiesはモデルに含まれる作業名をキー，作業インスタンスを値とした辞書である．
- modesはモデルに含まれるモード名をキー，モードインスタンスを値とした辞書である．
- resourcesはモデルに含まれる資源名をキー，資源インスタンスを値とした辞書である．
- temporalsはモデルに含まれる時間制約の先行作業名と後続作業名のタプルをキー，時間制約インスタンスを値とした辞書である．
- Paramsは最適化をコントロールするためのパラメータインスタンスである．
- Statusは最適化の状態を表す整数である．状態の種類と意味を，以下の表に示す．

最適化の状態を表す整数と意味

|  状態の定数   |  説明  |
| ---- | ---- |
| $-1$ |   実行不能（時間制約を満たす解が存在しない場合など） |
| $0$  |  最適化成功 |
| $7$  | 実行ファイルoptseq.exeのよび出しに失敗した． |
| $10$ | モデルの入力は完了しているが，まだ最適化されていない． |

```{.python.marimo}
class Model(BaseModel):
    """
    OptSeq model class.
        - Attributes:
            - activities: Dictionary that maps activity names to activity objects in the model.
            - modes: Dictionary that maps mode names to mode objects in the model.
            - resources:  Dictionary that maps resource names to resource objects in the model.
            - temporals: Dictionary that maps pairs of activity names to temporal constraint objects in the model.
            - Params: Object including all the parameters of the model.

            - act: List of all the activity objects in the model.
            - res: List of all the resource objects in the model.
            - tempo: List of all the tamporal constraint objects in the model.
    """
    name: Optional[str] = ''
    activities: Optional[Dict[str, Activity]] = {}
    modes: Optional[Dict[str, Mode]] = {}
    resources: Optional[Dict[str, Resource]] = {}
    temporals: Optional[Dict[str, Temporal]] = {}
    states: Optional[Dict[str, State]] = {}
    act: Optional[List[Activity]] = []
    res: Optional[List[Resource]] = []
    tempo: Optional[List[Temporal]] = []
    state: Optional[List[State]] = []
    Params: Optional[Parameters] = Parameters()
    Status: Optional[int] = 10
    ObjVal: Optional[NonNegativeInt] = None

    def __init__(self, name: Optional[str]=''):
        super().__init__(name=name)

    def __str__(self):
        ret = ['Model:{0}'.format(self.name)]
        ret.append('number of activities= {0}'.format(len(self.act)))
        ret.append('number of resources= {0}'.format(len(self.res)))
        if len(self.res):
            ret.append('\nResource Information')
            for res in self.res:
                ret.append(str(res))
                if len(res.terms) > 0:
                    ret.append(res.printConstraint())
        for a in self.act:
            for m in a.modes:
                self.modes[m.name] = m
        if len(self.modes):
            ret.append('\nMode Information')
            for i in self.modes:
                ret.append(str(i))
                ret.append(str(self.modes[i]))
        if len(self.act):
            ret.append('\nActivity Information')
            for act in self.act:
                ret.append(str(act))
        if len(self.tempo):
            ret.append('\nTemporal Constraint Information')
            for t in self.tempo:
                ret.append(str(t))
        if len(self.state):
            ret.append('\nState Information')
            for s in self.state:
                ret.append(str(s))
        return '\n'.join(ret)

    def addActivity(self, name='', duedate='inf', backward=False, weight=1, autoselect=False, quadratic=False):
        """
        Add an activity to the model.

            - Arguments:
                - name: Name for new activity. A string object except "source" and "sink." Remark that strings in OptSeq are restricted to a-z, A-Z, 0-9,[],_ and @.
                - duedate(optional): Duedate of activity. A non-nagative integer or string "inf."
                - backward(optional): True if activity is distached backwardly, False (default) otherwise.
                - weight(optional): Panalty of one unit of tardiness. Positive integer.
                - autoselect(optional): True or False flag that indicates the activity selects the mode automatically or not.

            - Return value: New activity object.

            - Example usage:

            >>> a = model.addActivity("act1")

            >>> a = model.addActivity(name="act1",duedate=20,weight=100)

            >>> a = model.addActivity("act1",20,100)
        """
        activity = Activity(name, duedate, backward, weight, autoselect, quadratic)
        self.act.append(activity)
        return activity

    def addResource(self, name='', capacity=None, rhs=0, direction='<=', weight='inf'):
        """
        Add a resource to the model.

            - Arguments:
                - name: Name for new resource. Remark that strings in OptSeq are restricted to a-z, A-Z, 0-9,[],_ and @.
                - capacity (optional): Capacity dictionary of the renewable (standard) resource.
                - Capacity dictionary maps intervals (pairs of start time and finish time) to amounts of capacity.
                - rhs (optional): Right-hand-side constant of nonrenewable resource constraint.
                - direction (optional): Rirection (or sense) of nonrenewable resource constraint; "<=" (default) or ">=" or "=".
                - weight (optional): Weight of resource. Non-negative integer or "inf" (default).

            - Return value: New resource object.

            - Example usage:

            >>> r=model.addResource("res1")

            >>> r=model.addResource("res1", {(0,10):1,(12,100):2} )

            >>> r=model.addResource("res2",rhs=10,direction=">=")

        """
        if capacity is None:
            capacity = {}
        res = Resource(name=name, capacity=capacity, rhs=rhs, direction=direction, weight=weight)
        self.res.append(res)
        return res

    def addTemporal(self, pred, succ, tempType='CS', delay=0, pred_mode=None, succ_mode=None):
        """
        Add a temporal constraint to the model.

        A temporal constraint has the following form::

            predecessor's completion (start) time +delay <=
                            successor's start (completion) time.

        Parameter "delay" can be negative.

            - Arguments:
                - pred: Predecessor (an activity object) or string "source."
                        Here, "source" specifies a dummy activity that precedes all other activities and starts at time 0.
                - succ: Successor (an activity object) or string "source."
                        Here, "source" specifies a dummy activity that precedes all other activities and starts at time 0.
                - tempType (optional): String that differentiates the temporal type.
                    "CS" (default)=Completion-Start, "SS"=Start-Start,
                    "SC"= Start-Completion, "CC"=Completion-Completion.
                - delay (optional): Time lag between the completion (start) times of two activities.
                - pred_mode (optional): Predecessor's mode
                - succ_mode (optional): Successor's mode

            - Return value: New temporal object.

            - Example usage:

            >>> t=model.addTemporal(act1,act2)

            >>> t=model.addTemporal(act1,act2,type="SS",delay=-10)

            To specify the start time of activity act is exactly 50, we use two temporal constraints:

            >>> t=model.addTemporal("source",act,type="SS",delay=50)

            >>> t=model.addTemporal(act,"source",type="SS",delay=50)
        """
        t = Temporal(pred, succ, tempType, delay, pred_mode, succ_mode)
        self.tempo.append(t)
        return t

    def addState(self, name=''):
        """
        Add a state to the model.

            - Arguments:
                - name: Name for new state. Remark that strings in OptSeq are restricted to a-z, A-Z, 0-9,[],_ and @.

            - Return value: New state object.

            - Example usage:

            >>> a = model.addState("state1")

        """
        s = State(name)
        self.state.append(s)
        return s

    def update(self):
        """
        prepare a string representing the current model in the OptSeq input format
        """
        makespan = self.Params.Makespan
        f = []
        self.resources = {}
        for r in self.res:
            self.resources[r.name] = r
            f.append(str(r))
        self.states = {}
        for s in self.state:
            self.states[s.name] = s
            f.append(str(s))
        self.modes = {}
        for a in self.act:
            for m in a.modes:
                self.modes[m.name] = m
        for m in self.modes:
            f.append('mode {0} '.format(m))
            f.append(str(self.modes[m]))
        self.activities = {}
        for a in self.act:
            self.activities[a.name] = a
            f.append(str(a))
        self.temporals = {}
        for t in self.tempo:
            if t.pred == 'source':
                pred = 'source'
            elif t.pred == 'sink':
                pred = 'sink'
            else:
                pred = t.pred.name
            if t.succ == 'source':
                succ = 'source'
            elif t.succ == 'sink':
                succ = 'sink'
            else:
                succ = t.succ.name
            self.temporals[pred, succ] = t
            f.append(str(t))
        for r in self.res:
            self.resources[r.name] = r
            if len(r.terms) > 0:
                f.append(r.printConstraint())
        if makespan:
            f.append('activity sink duedate 0 \n')
        return ' \n'.join(f)

    def optimize(self, cloud=False, init_fn='optseq_best_act_data.txt', best_fn='optseq_best_act_data.txt'):
        """
        Optimize the model using optseq.exe in the same directory.

            - Example usage:

            >>> model.optimize()
        """
        LOG = self.Params.OutputFlag
        f = self.update()
        if cloud:
            input_file_name = f'optseq_input{dt.datetime.now().timestamp()}.txt'
            f2 = open(input_file_name, 'w')
            p = pathlib.Path('.')
            script = p / 'scripts/optseq'
        else:
            f2 = open('optseq_input.txt', 'w')
            script = './optseq'
        f2.write(f)
        f2.close()
        import subprocess
        if platform.system() == 'Windows':
            cmd = f'optseq '
        elif platform.system() == 'Darwin':
            if platform.mac_ver()[2] == 'arm64':
                cmd = f'{script}-m1 '
            else:
                cmd = f'{script}-mac '
        elif platform.system() == 'Linux':
            cmd = f'{script}-linux '
            p = pathlib.Path('.')
        else:
            print(platform.system(), 'may not be supported.')
        cmd = cmd + ('-time ' + str(self.Params.TimeLimit) + ' -backtrack  ' + str(self.Params.Backtruck) + ' -iteration  ' + str(self.Params.MaxIteration) + ' -report     ' + str(self.Params.ReportInterval) + ' -seed      ' + str(self.Params.RandomSeed) + ' -tenure    ' + str(self.Params.Tenure) + ' -neighborhood   ' + str(self.Params.Neighborhood))
        if self.Params.Initial:
            cmd = cmd + f' -initial {init_fn}'
        try:
            if platform.system() == 'Windows':
                pipe = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            else:
                pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            if LOG:
                print('\n ================ Now solving the problem ================ \n')
            out, err = pipe.communicate(f.encode())
            if out == b'':
                print('error: could not execute command')
                print('please check that the solver is in the path')
                f2 = open('optseq_error.txt', 'w')
                f2.write('error: could not execute command')
                f2.close()
                self.Status = 7
                return
        except OSError:
            print('error: could not execute command')
            print('please check that the solver is in the path')
            f2 = open('optseq_error.txt', 'w')
            f2.write('error: could not execute command')
            f2.close()
            self.Status = 7
            return
        if cloud:
            os.remove(input_file_name)
        if int(sys.version_info[0]) >= 3:
            out = str(out, encoding='utf-8')
        if LOG == 2:
            print('\noutput:')
            print(out)
        if LOG:
            print('\nSolutions:')
        '\n        optseq output file\n        '
        if cloud:
            pass
        else:
            f3 = open('optseq_output.txt', 'w')
            f3.write(out)
            f3.close()
        infeasible = out.find('no feasible schedule found')
        if infeasible > 0:
            print('infeasible solution')
            self.Status = -1
            return
        self.Status = 0
        s0 = '--- best solution ---'
        s1 = '--- tardy activity ---'
        s2 = '--- resource residuals ---'
        s3 = '--- best activity list ---'
        s4 = 'objective value ='
        pos0 = out.find(s0) + len(s0)
        pos1 = out.find(s1, pos0)
        pos2 = out.find(s2, pos1)
        pos3 = out.find(s3, pos2)
        pos4 = out.find(s4, pos3)
        data = out[pos0:pos1]
        resdata = out[pos2 + len(s2):pos3]
        data = data.splitlines()
        reslines = resdata.splitlines()
        remain_data = out[pos4:].split()
        self.ObjVal = int(remain_data[3])
        bestactdata = out[pos3 + len(s3):pos4]
        if cloud:
            pass
        else:
            f3 = open(best_fn, 'w')
            f3.write(bestactdata.lstrip())
            f3.close()
        for line in reslines:
            if len(line) <= 1:
                continue
            current = line.split()
            resname = current[0][:-1]
            residual = current[1:]
            count = 0
            resDic = {}
            while count < len(residual):
                interval = residual[count].split(',')
                int1 = int(interval[0][1:])
                int2 = int(interval[1][:-1])
                count = count + 1
                num = int(residual[count])
                count = count + 1
                resDic[int1, int2] = num
            self.resources[resname].residual = resDic
        execute = []
        for i in range(len(data)):
            replaced = data[i].replace(',', ' ')
            current = replaced.split()
            if len(current) > 1:
                execute.append(current)
        for line in execute:
            actname = line[0]
            mode = line[1]
            try:
                start = line[2]
            except:
                print('Problem is infeasible')
                self.Status = -1
                return
            execute = line[3:-1]
            completion = line[-1]
            if LOG:
                print('{0:>10} {1:>5} {2:>5} {3:>5}'.format(actname, mode, start, completion))
            if actname == 'source':
                pass
            elif actname == 'sink':
                pass
            else:
                self.activities[actname].start = int(start)
                self.activities[actname].completion = int(completion)
                if mode != '---':
                    self.activities[actname].selected = self.modes[mode]
                else:
                    self.activities[actname].selected = self.activities[actname].modes[0]
                exeDic = {}
                for exe in execute:
                    exedata = exe.split('--')
                    start = exedata[0]
                    completion = exedata[1]
                    idx = completion.find('[')
                    if idx > 0:
                        parallel = completion[idx + 1:-1]
                        completion = completion[:idx]
                    else:
                        parallel = 1
                    exeDic[int(start), int(completion)] = int(parallel)
                self.activities[actname].execute = exeDic
        return

    def write(self, filename='optseq_chart.txt'):
        """
        Output the gantt's chart as a text file.

            - Argument:
                - filename: Output file name. Default="optseq_chart.txt."

            - Example usage:

            >>> model.write("sample.txt")

        """
        f = open(filename, 'w')
        horizon = 0
        actList = []
        for a in self.activities:
            actList.append(a)
            act = self.activities[a]
            horizon = max(act.completion, horizon)
        actList.sort()
        title = ' activity    mode'.center(20) + ' duration '
        width = len(str(horizon))
        for t in range(horizon):
            num = str(t + 1)
            title = title + (num.rjust(width) + '')
        f.write(title + '\n')
        f.write('-' * (30 + (width + 1) * horizon) + '\n')
        for a in actList:
            act = self.activities[a]
            actstring = act.name.center(10)[:10]
            if len(act.modes) >= 2 and act.selected.name is not None:
                actstring = actstring + str(act.selected.name).center(10)
                actstring = actstring + str(self.modes[act.selected.name].duration).center(10)
            else:
                actstring = actstring + str(act.modes[0].name).center(10)[:10]
                actstring = actstring + str(act.modes[0].duration).center(10)
            execute = [0 for t in range(horizon)]
            for s, c in act.execute:
                para = act.execute[s, c]
                for t in range(s, c):
                    execute[t] = int(para)
            for t in range(horizon):
                if execute[t] >= 2:
                    actstring = actstring + ('*' + str(execute[t]).rjust(width - 1))
                elif execute[t] == 1:
                    actstring = actstring + ('' + '=' * width)
                elif t >= act.start and t < act.completion:
                    actstring = actstring + ('' + '.' * width)
                else:
                    actstring = actstring + ('' + ' ' * width)
            actstring = actstring + ''
            f.write(actstring + '\n')
        f.write('-' * (30 + (width + 1) * horizon) + '\n')
        f.write('resource usage/capacity'.center(30) + ' \n')
        f.write('-' * (30 + (width + 1) * horizon) + '\n')
        resList = []
        for r in self.resources:
            resList.append(r)
        resList.sort()
        for r in resList:
            res = self.resources[r]
            if len(res.terms) == 0:
                rstring = res.name.center(30)
                cap = [0 for t in range(horizon)]
                residual = [0 for t in range(horizon)]
                for s, c in res.residual:
                    amount = res.residual[s, c]
                    if c == 'inf':
                        c = horizon
                    s = min(s, horizon)
                    c = min(c, horizon)
                    for t in range(s, c):
                        residual[t] = residual[t] + amount
                for s, c in res.capacity:
                    amount = res.capacity[s, c]
                    if c == 'inf':
                        c = horizon
                    s = min(s, horizon)
                    c = min(c, horizon)
                    for t in range(s, c):
                        cap[t] = cap[t] + amount
                for t in range(horizon):
                    num = str(cap[t] - residual[t])
                    rstring = rstring + ('' + num.rjust(width))
                f.write(rstring + '\n')
                rstring = str(' ').center(30)
                for t in range(horizon):
                    num = str(cap[t])
                    rstring = rstring + ('' + num.rjust(width))
                f.write(rstring + '\n')
                f.write('-' * (30 + (width + 1) * horizon) + '\n')
        f.close()

    def writeExcel(self, filename='optseq_chart.csv', scale=1):
        """
        Output the gantt's chart as a csv file for printing using Excel.

            - Argument:
                - filename: Output file name. Default="optseq_chart.csv."

            - Example usage:

            >>> model.writeExcel("sample.csv")

        """
        f = open(filename, 'w')
        horizon = 0
        actList = []
        for a in self.activities:
            actList.append(a)
            act = self.activities[a]
            horizon = max(act.completion, horizon)
        if scale <= 0:
            print('optseq write scale error')
            exit(0)
        original_horizon = horizon
        horizon = int(horizon / scale) + 1
        actList.sort()
        title = ' activity ,   mode,'.center(20) + ' duration,'
        width = len(str(horizon))
        for t in range(horizon):
            num = str(t + 1)
            title = title + (num.rjust(width) + ',')
        f.write(title + '\n')
        for a in actList:
            act = self.activities[a]
            actstring = act.name.center(10)[:10] + ','
            if len(act.modes) >= 2:
                actstring = actstring + (str(act.selected.name).center(10) + ',')
                actstring = actstring + (str(self.modes[act.selected.name].duration).center(10) + ',')
            else:
                actstring = actstring + (str(act.modes[0].name).center(10)[:10] + ',')
                actstring = actstring + (str(act.modes[0].duration).center(10) + ',')
            execute = [0 for t in range(horizon)]
            for s, c in act.execute:
                para = act.execute[s, c]
                for t in range(s, c):
                    t2 = int(t / scale)
                    execute[t2] = int(para)
            for t in range(horizon):
                if execute[t] >= 2:
                    actstring = actstring + ('*' + str(execute[t]).rjust(width - 1) + ',')
                elif execute[t] == 1:
                    actstring = actstring + ('' + '=' * width + ',')
                elif t >= int(act.start / scale) and t < int(act.completion / scale):
                    actstring = actstring + ('' + '.' * width + ',')
                else:
                    actstring = actstring + ('' + ' ' * width + ',')
            f.write(actstring + '\n')
        resList = []
        for r in self.resources:
            resList.append(r)
        resList.sort()
        for r in resList:
            res = self.resources[r]
            if len(res.terms) == 0:
                rstring = res.name.center(30) + ', , ,'
                cap = [0 for t in range(horizon)]
                residual = [0 for t in range(horizon)]
                for s, c in res.residual:
                    amount = res.residual[s, c]
                    if c == 'inf':
                        c = horizon
                    s = min(s, original_horizon)
                    c = min(c, original_horizon)
                    s2 = int(s / scale)
                    c2 = int(c / scale)
                    for t in range(s2, c2):
                        residual[t] = residual[t] + amount
                for s, c in res.capacity:
                    amount = res.capacity[s, c]
                    if c == 'inf':
                        c = horizon
                    s = min(s, original_horizon)
                    c = min(c, original_horizon)
                    s2 = int(s / scale)
                    c2 = int(c / scale)
                    for t in range(s2, c2):
                        cap[t] = cap[t] + amount
                for t in range(horizon):
                    rstring = rstring + (str(residual[t]) + ',')
                f.write(rstring + '\n')
        f.close()
```

### Modelクラスの使用例

```{.python.marimo}
model = Model()
_duration = {1: 13, 2: 25, 3: 15, 4: 27, 5: 22}
_act = {}
_mode = {}
_res = model.addResource('worker', capacity=1)
for _i in _duration:
    _act[_i] = model.addActivity(f'Act[{_i}]')
    _mode[_i] = Mode(f'Mode[{_i}]', _duration[_i])
    _mode[_i].addResource(_res, requirement=1)
    _act[_i].addModes(_mode[_i])
model.addTemporal(_act[1], _act[2], delay=20)
model.addTemporal(_act[1], _act[3], delay=20)
model.addTemporal(_act[2], _act[4], delay=10)
model.addTemporal(_act[2], _act[5], delay=8)
model.addTemporal(_act[3], _act[4], delay=10)
model.addTemporal('source', _act[1], delay=5)
model.addTemporal(_act[4], 'sink', delay=5)
model.Params.TimeLimit = 1
model.Params.Makespan = True
model.optimize(cloud=False)
```

## Graphvizによる可視化関数 visualize

<img src="https://github.com/scmopt/scmopt_data/blob/main/fig/optseq-graphviz.png?raw=true" width=800 height=200>

```{.python.marimo}
def visualize(model):
    g = graphviz.Digraph('G', filename='optseq.gv')
    g.graph_attr['rankdir'] = 'LR'
    for r, res in model.resources.items():
        g.node(name=r, shape='trapezium', color='green')
    for a, act in model.activities.items():
        with g.subgraph(name=f'cluster[{a}]') as c:
            c.attr(style='filled', color='lightgrey')
            c.node(name=a, shape='rectangle', color='red')
            for mode in act.modes:
                c.node(name=mode.name, shape='box3d', color='blue')
                c.edge(a, mode.name, arrowhead='tee', style='dotted', color='blue')
                for rname, rtype in mode.requirement:
                    g.edge(mode.name, rname, arrowhead='box', label=rtype, style='dashed', color='green')
    for t, temp in model.temporals.items():
        if temp.pred == 'source':
            pred = 'source'
        elif temp.pred == 'sink':
            pred = 'sink'
        else:
            pred = temp.pred.name
        if temp.succ == 'source':
            succ = 'source'
        elif temp.succ == 'sink':
            succ = 'sink'
        else:
            succ = temp.succ.name
        if pred == 'source' or pred == 'sink':
            g.node(name=pred, shape='oval', color='red')
        if succ == 'source' or succ == 'sink':
            g.node(name=succ, shape='oval', color='red')
        if temp.type != 'CS':
            label = temp.type
        else:
            label = ''
        if temp.delay > 0:
            label = label + f'({temp.delay})'
        g.edge(pred, succ, arrowhead='open', label=label)
    return g
```

```{.python.marimo}
g = visualize(model)
#g.view();
```

## 最適化の描画関数 plot_optseq

OptSeqはメタヒューリスティクスによって解の探索を行う．
一般には，解の良さと計算時間はトレードオフ関係がある．つまり，計算時間をかければかけるほど良い解を得られる可能性が高まる．
どの程度の計算時間をかければ良いかは，最適化したい問題例（問題に数値を入れたもの）による．
plot_optseqは，横軸に計算時間，縦軸に目的関数値をプロットする関数であり，最適化を行ったあとに呼び出す．
得られるPlotlyの図は，どのくらいの計算時間をかければ良いかをユーザーが決めるための目安を与える．

たとえば以下の例の図から，10秒程度の計算時間で良好な解を得ることができるが，念入りにさらに良い解を探索したい場合には30秒以上の計算時間が必要なことが分かる．

<img src="https://github.com/scmopt/scmopt_data/blob/main/fig/plot-optseq.png?raw=true" width=600 height=200>

```{.python.marimo}
# | export
def plot_optseq(file_name: str = "optseq_output.txt"):
    with open(file_name) as f:
        out = f.readlines()
    x, y = [], []
    for l in out[7:]:
        sep = re.split("[=()/]", l)
        # print(sep)
        if sep[0] == "--- best solution ---\n":
            break
        if sep[0] == "objective value ":
            val, cpu = map(float, [sep[1], sep[3]])
            x.append(cpu)
            y.append(val)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+lines",
            name="value",
            marker=dict(size=10, color="black"),
        )
    )
    fig.update_layout(
        title="OptSeq performance", xaxis_title="CPU time", yaxis_title="Value"
    )
    return fig
```

```{.python.marimo}
_fig = plot_optseq()
#plotly.offline.plot(_fig);
```

## ガントチャートを生成する関数 make_gantt

与えられたモデルに格納されているスケジュールを、ガントチャートで可視化する。
ただし，一部分を描画したい場合には，開始期間beginと終了期間endならびに資源名の集合resourcesで制御する．

引数：

- model : OptSeqモデルオブジェクト
- start : 開始時刻を表す（pandasの日付時刻型に変換可能な文字列）文字列． 既定値は "2024/1/1"．
- period : 時間の単位を表す文字列． "days"（日）， "seconds"（秒），　"minutes"（分）， "hours（時間）の何れか． 既定値は "days"
- begin: 開始期間（整数）；既定値は $0$
- end: 終了期間（整数もしくは"inf"）；既定値は "inf"
- resources: 資源名の集合；既定値はNone

返値：

- fig : ガントチャートの図オブジェクト

```{.python.marimo}
# | export
def time_convert_long(periods, start="2019/1/1", period="days"):
    start = pd.to_datetime(start)
    if period == "days":
        time_ = start + dt.timedelta(days=float(periods))
    elif period == "hours":
        time_ = start + dt.timedelta(hours=float(periods))
    elif period == "minutes":
        time_ = start + dt.timedelta(minutes=float(periods))
    elif period == "seconds":
        time_ = start + dt.timedelta(seconds=float(periods))
    else:
        raise TypeError("pariod must be 'days' or 'seconds' or minutes' or 'days'")
    return time_.strftime("%Y-%m-%d %H:%M:%S")
```

```{.python.marimo}
# | export
def make_gantt(
    model: Model,
    start: str = "2024/1/1",
    period: str = "days",
    begin: int = 0,
    end: Union[int, str] = "inf",
    resources: Optional[Set] = None,
) -> plotly.graph_objs._figure.Figure: 
    """
    ガントチャートを生成する関数
    """
    if resources is None:  # 制限なし
        resources = []
        for res in model.res:
            resources.append(res.name)
        resources = set(resources)

    if end == "inf":
        end = 0
        for a in model.act:
            end = max(a.completion, end)

    # 資源ごとに区間木を準備する
    interval_tree = {}
    for r in resources:
        # res = model.resources[r]
        interval_tree[r] = IntervalTree()

    for i in model.activities:
        a = model.activities[i]
        if a.selected is not None:
            m = a.selected  # mode selected
            for r, _ in m.requirement:
                if r in resources:
                    interval_tree[r][a.start : a.completion] = a

    L = []
    for r in resources:
        res = model.resources[r]
        for interval in interval_tree[r][begin:end]:  # 資源ごとの区間木 => 区間オブジェクトを返す
            act = interval.data
            st = time_convert_long(interval.begin, start=start, period=period)
            fi = time_convert_long(interval.end, start=start, period=period)
            L.append(dict(Task=act.name, Start=st, Finish=fi, Resource=r))

    df = pd.DataFrame(L)
    fig = px.timeline(
        df, x_start="Start", x_end="Finish", y="Resource", color="Task", opacity=0.5
    )
    return fig
```

### make_gantt関数の使用例


<img src="https://github.com/scmopt/scmopt_data/blob/main/fig/make_gannt.png?raw=true" width=600 height=200>

```{.python.marimo}
#modelインスタンスは生成済み
_begin = 0
_end = 50000
_resources = None #set(["worker"])
_fig = make_gantt(model, start="2024-1-1", period="minutes", begin=_begin, end=_end, resources=_resources)
#plotly.offline.plot(_fig);
```

## 資源の占有率を表示する関数 make_resource_usage

```{.python.marimo}
def make_resource_usage(model: Model, begin: int=0, end: Union[int, str]='inf', width: int=1000, resources: Optional[Set]=None) -> Tuple[plotly.graph_objs._figure.Figure, pd.DataFrame]:
    if resources is None:
        resources = []
        for res in model.res:
            resources.append(res.name)
        resources = set(resources)
    if end == 'inf':
        end = 0
        for a in model.act:
            end = max(a.completion, end)
    interval_tree, interval_tree_rest = ({}, {})
    for r in resources:
        interval_tree[r] = IntervalTree()
        interval_tree_rest[r] = IntervalTree()
    for i in model.activities:
        a = model.activities[i]
        if a.selected is not None:
            m = a.selected
            for r, _ in m.requirement:
                if r in resources:
                    interval_tree[r][a.start:a.completion] = a
    for r in resources:
        t = 0
        for st, fi in model.resources[r].capacity:
            cap = model.resources[r].capacity[st, fi]
            if fi == 'inf':
                fi = end
            interval_tree_rest[r][st:fi] = cap
    total_duration = defaultdict(int)
    total_capacity = defaultdict(int)
    start = begin
    finish = start + width
    while finish < end:
        for r in resources:
            for interval in interval_tree[r][start:finish]:
                act = interval.data
                total_duration[r, start, finish] = total_duration[r, start, finish] + (act.completion - act.start)
                total_duration[r, start, finish] = total_duration[r, start, finish] - max(0, act.completion - finish)
                total_duration[r, start, finish] = total_duration[r, start, finish] - max(0, start - act.start)
            for interval in interval_tree_rest[r][start:finish]:
                cap = interval.data
                total_capacity[r, start, finish] = total_capacity[r, start, finish] + (interval.end - interval.begin) * cap
                total_capacity[r, start, finish] = total_capacity[r, start, finish] - max(0, (interval.end - finish) * cap)
                total_capacity[r, start, finish] = total_capacity[r, start, finish] - max(0, (start - interval.begin) * cap)
        start = start + width
        finish = start + width
    idx, ratio, resource = ([], [], [])
    for r, start, finish in total_capacity:
        ratio.append(total_duration[r, start, finish] / total_capacity[r, start, finish])
        resource.append(r)
        idx.append(start)
    df = pd.DataFrame({'idx': idx, 'ratio': ratio, 'resource': resource})
    fig = px.line(df, x='idx', y='ratio', color='resource')
    return (fig, df)
```

### make_resource_usage関数の使用例

```{.python.marimo}
_begin = 0
_end = 'inf'
_width = 100
_resources = set(["worker"])
_fig, _df = make_resource_usage(model, begin=_begin, end=_end, width=_width, resources=_resources)
#plotly.offline.plot(_fig)
```

## 資源グラフを生成する関数　make_resource_graph

- model: OptSeqモデルファイル
- start : 開始時刻を表す（pandasの日付時刻型に変換可能な文字列）文字列． 既定値は "2024/1/1"．
- period : 時間の単位を表す文字列． "days"（日）， "seconds"（秒），　"minutes"（分）， "hours（時間）の何れか． 既定値は "days"
- scale: 横軸をscale分の1にする．（たとえば分単位を時間単位にする．）既定値は $1$．
- begin: 開始期間（整数）；既定値は $0$
- end: 終了期間（整数もしくは"inf"）；既定値は "inf"
- resources: 資源名の集合；既定値はNo

返値：

- fig : 資源グラフの図オブジェクト

```{.python.marimo}
def make_resource_graph(model: Model, start: str='2024/1/1', period: str='days', 
                        scale: int=1, begin: int=0, end: Union[int, str]='inf', resources: Optional[Set]=None):
    """
    資源の使用量と残差（容量-使用量）を図示する関数
    """
    if resources is None:
        resources = []
        for res in model.res:
            resources.append(res.name)
        resources = set(resources)
    if end == 'inf':
        end = 0
        for a in model.act:
            end = max(a.completion, end)
    horizon = end - begin
    count = 0
    resource_list = []
    for r in resources:
        res = model.resources[r]
        if len(res.terms) == 0:
            count = count + 1
            resource_list.append(res.name)
    if count >= 1:
        fig = make_subplots(rows=count, cols=1, subplot_titles=resource_list)
    else:
        fig = {}
    for count, r in enumerate(resource_list):
        res = model.resources[r]
        cap = defaultdict(int)
        residual = defaultdict(int)
        usage = defaultdict(int)
        for s, c in res.residual:
            amount = res.residual[s, c]
            if c == 'inf':
                c = end
            if c <= begin:
                continue
            if end <= s:
                break
            s = min(s, end)
            c = min(c, end)
            for t in range(s, c):
                residual[t] = residual[t] + amount
        for s, c in res.capacity:
            amount = res.capacity[s, c]
            if c == 'inf':
                c = end
            if c <= begin:
                continue
            if end <= s:
                break
            s = min(s, end)
            c = min(c, end)
            for t in range(s, c):
                cap[t] = cap[t] + amount
        for t in range(begin, end):
            usage[t] = cap[t] - residual[t]
        usage_list, residual_list = ([], [])
        for t in range(begin, end):
            usage_list.append(usage[t])
            residual_list.append(residual[t])
        horizon2 = int(horizon / scale)
        t = 0
        usage2 = []
        residual2 = []
        for i in range(horizon2):
            average_usage = 0
            average_residual = 0
            for j in range(scale):
                average_usage = average_usage + usage_list[t]
                average_residual = average_residual + residual_list[t]
                t = t + 1
            usage2.append(average_usage / scale)
            residual2.append(average_residual / scale)
        t = 0
        x = []
        for i in range(horizon2):
            x.append(time_convert_long(t, start=start, period=period))
            t = t + scale
        fig.add_trace(go.Bar(name='Usage', x=x, y=usage2, marker_color='crimson'), row=count + 1, col=1)
        fig.add_trace(go.Bar(name='Residual', x=x, y=residual2, marker_color='lightslategrey'), row=count + 1, col=1)
    fig.update_layout(barmode='stack', title_text=f'Capacity/Usage', showlegend=False)
    return fig
```

### make_resource_graph関数の使用例


<img src="https://github.com/scmopt/scmopt_data/blob/main/fig/make_resource_graph.png?raw=true" width=600 height=200>

```{.python.marimo}
_start: str = '2024/1/1'
_period: str = 'minutes'
_scale = 10
_begin = 10
_end = 1000
_resources = None
_fig = make_resource_graph(model, start=_start, period=_period, scale=_scale, begin= _begin, end= _end, resources= _resources)
#plotly.offline.plot(_fig);
```

<!-- ## Large Benchmark Instances

https://link.springer.com/chapter/10.1007/978-3-030-30048-7_9

https://arxiv.org/pdf/2102.08778v2.pdf

https://arxiv.org/pdf/1909.08247.pdf

benchmarks:

https://drive.google.com/drive/folders/1QuKEABR9aiNKPIFe0VMFXP7BNor8KW9b

相対計算時間は600秒での反復数から計算

| 問題サイズ  | 相対計算時間  | メモリ (Scaled) |
|-----------|---------|-------|
| 10,10     | 1       | 2.8 MB (1)  |
| 100,10    | 10      | 54.2 MB (19) |
| 1000,10   | 592     | 3.1 GB (1110)|
| 100,100   | 53    　　 | 4.6GB (1300) |

- Solver (C++, Apache 2)

https://github.com/lpierezan/jssp-solver/tree/master -->

```{.python.marimo}
#| hide
def make_benchmark_model(name:str, time_limit:int=3600, slow_machine:int = -1, increase:float = 1.0) -> Model:

    fname = f"./data/optseq/{name}"


    f = open(fname, "r")
    lines = f.readlines()
    f.close()
    n, m = map(int, lines[0].split())
    #print("n,m=", n, m)

    xx = np.zeros(2*n*m, dtype=int)

    model = Model()
    act, mode, res = {}, {}, {}
    for j in range(m):
        res[j] = model.addResource(f"machine[{j}]", capacity=1)

    # prepare data as dic
    machine, proc_time = {}, {}
    for i in range(n):
        L = list(map(int, lines[i + 1].split()))
        for j in range(m):
            machine[i, j] = L[2 * j]
            proc_time[i, j] = L[2 * j + 1]

    #機械 slow_machine を increase倍遅くする
    if slow_machine>=0:
        for i in range(n):
            proc_time[i,slow_machine] = int(proc_time[i,slow_machine]*increase)

    for i in range(n):
        for j in range(m):
            xx[j+m*i] = machine[i, j] 
            xx[n*m+j+m*i] = proc_time[i, j]

    for i, j in proc_time:
        act[i, j] = model.addActivity(f"Act[{i},{j}]")
        mode[i, j] = Mode(f"Mode[{i}{j}]", proc_time[i, j])
        mode[i, j].addResource(res[machine[i,j]], 1)
        act[i, j].addModes(mode[i, j])

    for i in range(n):
        for j in range(m - 1):
            model.addTemporal(act[i, j], act[i, j + 1])

    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = False
    model.Params.Makespan = True

    return model, xx
```

```{.python.marimo}
#| hide
# model.optimize()
# fig = make_gantt(model, start="2024-1-1", period="minutes", begin=100, end=2000)
# plotly.offline.plot(fig);
# start:str = "2024/1/1"
# period:str ="minutes"
# scale = 100
# begin = 10
# end = 70000
# resources = set(["machine[1]","machine[2]"])
# fig = make_resource_graph(model, start=start, period=period, scale=scale, begin=begin, end=end, resources=resources )
# plotly.offline.plot(fig);
# fig = plot_optseq()
# plotly.offline.plot(fig);
```

## OptSeqのモデルをCPのモデルに変換する関数 transform_to_cp

ジョブショップスケジューリングの大規模問題例用に他のソルバーも準備する．ただし，より簡略化したモデルに対する最適化になる．
以下の関数は，機械（資源量上限が1の資源）を特別に扱うので， 大規模問題例ではOptSeqより高速で精度が良い可能性がある．
実務においては，様々なソルバーを準備しておき，すべてで試し， 最も性能が良いものを使う確率を上げていくAutoOptの利用が望ましい．

以下の関数では，CPにより最適化を行い結果をOptSeqのモデルインスタンスに書き込む． 引数のhorizonはなるべく小さい計画期間を事前に計算して設定しておくことが望ましい．
ただし，horizonが最適メイクスパンより小さいと実行不能になるので，注意する必要がある．
簡易計算用の関数 compute_horizon も準備しておくが， ベンチマーク問題例では大きくなりすぎるので，
別途簡易ヒューリスティクスなどで解いて設定する必要がある． ただし，メモリ量はあまり変わらない．

TODO: 後ろ詰めの実装， 時刻依存の資源量， 2次の納期遅れペナルティ，　．．．

引数

- model: OptSeqのモデルインスタンス
- horizon: 計画期間

返値

- cpmodel: CPのモデル

```{.python.marimo}
def compute_horizon(model: Model) -> int:
    horizon = 0
    for a in model.act:
        max_duration = 0
        for m in a.modes:
            max_duration = max(max_duration, m.duration)
        horizon = horizon + max_duration
    for r in model.res:
        t = 0
        for st, fi in r.capacity:
            if t < st:
                horizon = horizon + (st - t)
            t = fi
    return horizon
```

```{.python.marimo}
#| export
def transform_to_cp(model: Model, horizon:int):

    model.update()

    #容量１の資源は機械として別処理をする． termsがあるものは再生不能資源（絶対制約とする）
    machines = set([])
    resources = set([])
    nonrenewable = set([])

    #資源の分類
    for r in model.res:
        if len(r.terms) > 0: #再生不能資源
            nonrenewable.add(r.name)
        elif r.capacity == 1: #機械
            machines.add(r.name)
        else: #一般の資源
            max_cap = 0
            for (st,fi), cap in r.capacity.items():
                #print(st,fi, cap)
                max_cap = max(max_cap, cap)
            if max_cap==1:
                machines.add(r.name)
            else:
                resources.add(r.name)

    cpmodel = CpModel()

    start, end, act = {}, {}, {}
    selected = {} #複数モード用
    activities_on_machine = defaultdict(list)
    modes_for_act = defaultdict(list)

    all_ends = set([]) #メイクスパン計算用
    for a in model.act:
        start[a.name] = cpmodel.new_int_var(0, horizon, f'start({a.name})')
        end[a.name] = cpmodel.new_int_var(0, horizon, f'end({a.name})')

        for m in a.modes:
            duration = m.duration
            #複数モードの optional_interval_var
            selected[a.name,m.name] = cpmodel.new_bool_var(f'selected({a.name,m.name})') #モードの選択を表すブール変数
            act[a.name,m.name] = cpmodel.new_optional_interval_var(
                                  start[a.name], duration, end[a.name], selected[a.name,m.name],
                                  f'act({a.name,m.name})')
            modes_for_act[a.name].append( selected[a.name,m.name] )

            all_ends.add(end[a.name])

            if m.requirement is not None:
                for (r, _), req in m.requirement.items():
                    if r in machines: #機械 (add_no_overlap)
                        activities_on_machine[r].append( act[a.name,m.name] )
                    else:
                        #容量2以上の資源 (add_cumulative) 
                        activities_on_machine[r].append( (act[a.name,m.name], req) )

        cpmodel.add_exactly_one( modes_for_act[a.name] ) #１つのモードを選択

    #資源量上限が1の機械に対する制約
    for r in machines:
        #使用不能な時間帯にダミーの作業を配置
        t = 0
        dummy_acts = []
        for (st,fi) in model.resources[r].capacity:
            if t < st: #以前の終了から開始までの期間が休み
                dummy_acts.append( cpmodel.NewIntervalVar(t, st-t, st, f'machine_stop({r}_{st}_{fi})') )
            t = fi
        cpmodel.add_no_overlap( activities_on_machine[r]+dummy_acts )

    #機械以外の資源制約
    for r in resources:
        demand =[]
        for (_, req)  in activities_on_machine[r]:
            for (st,fi), dem in req.items():
                demand.append(dem)
                continue
        capacity = 0 
        for (st,fi), cap in model.resources[r].capacity.items():
            capacity = max(capacity, cap)
        #使用不能な時間帯にダミーの作業を配置
        t = 0
        dummy_acts, dummy_demands = [], []
        for (st,fi), cap in model.resources[r].capacity.items():
            if t < st: #最初の期間が休み
                dummy_acts.append( cpmodel.NewIntervalVar(t, st-t, st, f'machine_stop({r}_{st}_{fi})') )
                dummy_demands.append(capacity)
            if capacity -cap > 0: #不足分をダミーで配置
                dummy_acts.append( cpmodel.NewIntervalVar(st, fi-st, fi, f'machine_stop({r}_{st}_{fi})') )
                dummy_demands.append(capacity - cap)
            t = fi

        cpmodel.add_cumulative(intervals =[activity for (activity,_ )  in activities_on_machine[r]],
                               demands = demand, capacity = capacity)
    #再生不能資源
    constraint ={}
    for r in nonrenewable:
        res = model.resources[r]
        if res.direction == "<" or res.direction == "<=":
            constraint[r] = cpmodel.add(
                sum( coeff* selected[a.name,m.name] for (coeff, a, m) in res.terms ) <= int(res.rhs) )
        elif res.direction == "=" or res.direction == "==":
            constraint[r] = cpmodel.add(
                sum( coeff* selected[a.name,m.name] for (coeff, a, m) in res.terms ) == int(res.rhs) )
        elif res.direction == "=" or res.direction == ">=":
            constraint[r] = cpmodel.add(
                sum( coeff* selected[a.name,m.name] for (coeff, a, m) in res.terms ) >= int(res.rhs) )
        else:
            raise TypeError("constraint direction musr be <=, =, or >=")

    #時間制約（source,sink間の時間制約は考慮していない） => sourceは0, sinkはmakespanを追加すればできる
    #リリース時刻は開始時刻の範囲を限定した方が効率的なので，sourceとの時間制約ではなく，　別途処理する！
    for temp in model.tempo:
        if temp.pred_mode is None and temp.succ_mode is None: #モードに依存しない時間遅れをもつ時間制約
            if temp.type =="CS":
                cpmodel.add(end[temp.pred.name] + temp.delay <=start[temp.succ.name])
            elif temp.type =="CC":
                cpmodel.add(end[temp.pred.name] + temp.delay <=end[temp.succ.name])
            elif temp.type =="SS":
                cpmodel.add(start[temp.pred.name] + temp.delay <=start[temp.succ.name])
            elif temp.type =="SC":
                cpmodel.add(start[temp.pred.name] + temp.delay <=end[temp.succ.name])
        else: #モードに依存する時間遅れをもつ時間制約
            if temp.pred_mode is not None and temp.succ_mode is None: #先行ジョブのみモード依存
                if temp.type =="CS":
                    cpmodel.add(end[temp.pred.name] + temp.delay <=start[temp.succ.name]).OnlyEnforceIf(
                        selected[temp.pred.name,temp.pred_mode.name])
                elif temp.type =="CC":
                    cpmodel.add(end[temp.pred.name] + temp.delay <=end[temp.succ.name]).OnlyEnforceIf(
                        selected[temp.pred.name,temp.pred_mode.name])
                elif temp.type =="SS":
                    cpmodel.add(start[temp.pred.name] + temp.delay <=start[temp.succ.name]).OnlyEnforceIf(
                        selected[temp.pred.name,temp.pred_mode.name])
                elif temp.type =="SC":
                    cpmodel.add(start[temp.pred.name] + temp.delay <=end[temp.succ.name]).OnlyEnforceIf(
                        selected[temp.pred.name,temp.pred_mode.name])
            elif temp.pred_mode is None and temp.succ_mode is not None: #後続ジョブのみモード依存
                if temp.type =="CS":
                    cpmodel.add(end[temp.pred.name] + temp.delay <=start[temp.succ.name]).OnlyEnforceIf(
                        selected[temp.succ.name,temp.succ_mode.name])
                elif temp.type =="CC":
                    cpmodel.add(end[temp.pred.name] + temp.delay <=end[temp.succ.name]).OnlyEnforceIf(
                        selected[temp.succ.name,temp.succ_mode.name])
                elif temp.type =="SS":
                    cpmodel.add(start[temp.pred.name] + temp.delay <=start[temp.succ.name]).OnlyEnforceIf(
                        selected[temp.succ.name,temp.succ_mode.name])
                elif temp.type =="SC":
                    cpmodel.add(start[temp.pred.name] + temp.delay <=end[temp.succ.name]).OnlyEnforceIf(
                        selected[temp.succ.name,temp.succ_mode.name])
            else: #先行ジョブ，後続ジョブともにモード依存
                if temp.type =="CS":
                    cpmodel.add(end[temp.pred.name] + temp.delay <=start[temp.succ.name]).OnlyEnforceIf(
                        [selected[temp.pred.name,temp.pred_mode.name], selected[temp.succ.name,temp.succ_mode.name]] )
                elif temp.type =="CC":
                    cpmodel.add(end[temp.pred.name] + temp.delay <=end[temp.succ.name]).OnlyEnforceIf(
                        [selected[temp.pred.name,temp.pred_mode.name], selected[temp.succ.name,temp.succ_mode.name]] )
                elif temp.type =="SS":
                    cpmodel.add(start[temp.pred.name] + temp.delay <=start[temp.succ.name]).OnlyEnforceIf(
                        [selected[temp.pred.name,temp.pred_mode.name], selected[temp.succ.name,temp.succ_mode.name]] )
                elif temp.type =="SC":
                    cpmodel.add(start[temp.pred.name] + temp.delay <=end[temp.succ.name]).OnlyEnforceIf(
                        [selected[temp.pred.name,temp.pred_mode.name], selected[temp.succ.name,temp.succ_mode.name]] )

    if model.Params.Makespan: #メイクスパン
        obj = cpmodel.new_int_var(0, horizon, 'makespan')
        cpmodel.add_max_equality(obj, list(all_ends) )
        cpmodel.minimize(obj)
    else: #納期遅れ最小化
        delay = {}
        for a in model.act:
            delay[a.name] = cpmodel.new_int_var(0, horizon, f'delay({a.name})')
            if a.duedate=="inf":
                a.duedate = horizon
            cpmodel.add( delay[a.name] >= end[a.name]-a.duedate )
        cpmodel.minimize( sum( a.weight*delay[a.name] for a in model.act)  )

    # Solve model
    solver = CpSolver()
    solver.parameters.max_time_in_seconds = model.Params.TimeLimit 
    solver.parameters.log_search_progress = model.Params.OutputFlag

    status = solver.Solve(cpmodel)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE: 
        model.ObjVal = int(solver.ObjectiveValue())
        model.Status = status
        for a_name, a in model.activities.items():
            for m in a.modes:
                if solver.value(selected[a_name,m.name]):
                    a.selected = m
        for a_name, a in model.activities.items():
            a.start = solver.value(start[a_name])
            a.completion = solver.value(end[a_name])
            a.execute = {(a.start, a.completion): 1}
    elif status == cp_model.INFEASIBLE:
        model.Status = -1
    else:
        model.Status = status 

    return cpmodel
```

<!-- ### Modelのスケールによる近似

基本使用メモリはあまり変わらないが，計算速度はアップする．
探索中にメモリが増大することを防ぐことができるが， 作業時間の分布に依存する．

作業時間の小さいものを除いて最適化し，後で再挿入という方法も考えられる．
 -->

```{.python.marimo}
#| hide
# import copy
# import math
# name = "tai_j1000_m100_1.data"
# model = make_benchmark_model(name, 10000)

# model_ = copy.deepcopy(model) #backup
# scale = 10
# for a_name, a in model.activities.items():
#     for m in a.modes:
#         m.duration = math.ceil(m.duration/10)
#         #print(m.duration)
```

<!-- ### transform_to_cpの使用例

TODO: OptSeqのsolveのオプションで切り替える．

| 問題サイズ  | 計算時間  | メモリ  |
|-----------|---------|-------|
| 10,10     | 0.23    |  - |
| 100,10    | 0.83    |  - |
| 1000,10   | 49      | - |
| 100,100   | 102    　　| - |
| 1000,100   | 10000    | 32 GB+|
| 1000,1000  |   -     |   -  GB| -->

```{.python.marimo}
# model_1 = Model()
# duration_1 = {1: 13, 2: 25, 3: 15, 4: 27, 5: 22}
# act_3 = {}
# mode_3 = {}
# res_2 = model_1.addResource('worker', capacity=1)
# for i_1 in duration_1:
#     act_3[i_1] = model_1.addActivity(f'Act[{i_1}]')
#     mode_3[i_1, 1] = Mode(f'Mode[{i_1},{1}]', duration_1[i_1])
#     mode_3[i_1, 2] = Mode(f'Mode[{i_1},{2}]', duration_1[i_1])
#     mode_3[i_1, 1].addResource(res_2, requirement=1)
#     act_3[i_1].addModes(mode_3[i_1, 1], mode_3[i_1, 2])
# model_1.addTemporal(act_3[1], act_3[2], delay=20, pred_mode=mode_3[1, 1], succ_mode=mode_3[2, 1])
# model_1.addTemporal(act_3[1], act_3[2], delay=30, pred_mode=mode_3[1, 2], succ_mode=mode_3[2, 2])
# model_1.addTemporal(act_3[1], act_3[2], delay=40, pred_mode=mode_3[1, 1], succ_mode=mode_3[2, 2])
# model_1.addTemporal(act_3[1], act_3[2], delay=50, pred_mode=mode_3[1, 2], succ_mode=mode_3[2, 1])
# model_1.addTemporal(act_3[1], act_3[2], delay=20)
# model_1.addTemporal(act_3[1], act_3[3], delay=20)
# model_1.addTemporal(act_3[2], act_3[4], delay=10)
# model_1.addTemporal(act_3[2], act_3[5], delay=8)
# model_1.addTemporal(act_3[3], act_3[4], delay=10)
# model_1.Params.TimeLimit = 1
# model_1.Params.Makespan = True
# model_1.optimize()
```

```{.python.marimo}
# model_2 = Model()
# due = {1: 5, 2: 9, 3: 6, 4: 4}
# duration_2 = {1: 1, 2: 2, 3: 3, 4: 4}
# res_3 = model_2.addResource('writer')
# res_3.addCapacity(0, 'inf', 1)
# act_4 = {}
# mode_4 = {}
# for i_2 in duration_2:
#     act_4[i_2] = model_2.addActivity(f'Act[{i_2}]', duedate=due[i_2], weight=1, backward=False)
#     mode_4[i_2] = Mode(f'Mode[{i_2}]', duration_2[i_2])
#     mode_4[i_2].addResource(res_3, 1)
#     act_4[i_2].addModes(mode_4[i_2])
# model_2.Params.TimeLimit = 1
# model_2.Params.Makespan = False
# model_2.optimize()
```

```{.python.marimo}
# stop
# horizon = 1000000
# print('horizon=', horizon)
# model_2.Params.OutputFlag = False
# cpmodel = transform_to_cp(model_2, horizon)
# print(model_2.ObjVal, model_2.Status)
```

```{.python.marimo}
# import time
# import zlib
# import tracemalloc
# name = 'tai_j10_m10_1.data'
# model_3, xx = make_benchmark_model(name, 10000)
# model_3.Params.TimeLimit = 10
# horizon_1 = 1000000
# print('horizon=', horizon_1)
# model_3.Params.OutputFlag = False
# start_time = time_1.time()
# cpmodel_1 = transform_to_cp(model_3, horizon_1)
# end_time = time_1.time()
# print(model_3.ObjVal, model_3.Status)
```

<!-- ## MOAIアプローチ

機械学習でスケジューリングを高速化 -->

```{.python.marimo}
# np.random.seed(123)
# name_1 = 'tai_j100_m100_1.data'
# n = 100
# m = 100
# num_steps = 10
# max_iter = m * num_steps
# X = np.zeros((max_iter, 2 * n * m), dtype=int)
# y = np.zeros((max_iter, n * m), dtype=int)
# iter = 0
# models = []
# for slow_machine in range(m):
#     print('m=', slow_machine, end=' ')
#     for increase in np.linspace(1.0, 1.5, num_steps):
#         model_4, xx_1 = make_benchmark_model(name_1, time_limit=30, slow_machine=slow_machine, increase=increase)
#         cpmodel_2 = transform_to_cp(model_4, horizon_1)
#         if model_4.Status == cp_model.OPTIMAL or model_4.Status == cp_model.FEASIBLE:
#             models.append((model_4.ObjVal, slow_machine, increase))
#             yy = np.zeros(n * m, dtype=int)
#             for i_3, a in enumerate(model_4.act):
#                 yy[i_3] = a.start
#             X[iter] = xx_1
#             y[iter] = yy
#             iter = iter + 1
#         else:
#             print('failed')
```

```{.python.marimo}
# from sklearn.datasets import make_multilabel_classification
# import xgboost as xgb 

# X, y = make_multilabel_classification(
#     n_samples=32, n_classes=5, n_labels=3, random_state=0
# )
# clf = xgb.XGBClassifier(tree_method="hist")
# clf.fit(X, y)
# np.testing.assert_allclose(clf.predict(X), y)
```

```{.python.marimo}
#| hide
# from sklearn.metrics import accuracy_score, r2_score
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor


# #train-test split （自分で並べ替えを行うように変更）
# def train_test_split2(X, y, models, test_size=0.1):
#     np.random.seed(13)
#     max_iter = len(X)
#     n_train = int(max_iter*(1-test_size))
#     perm = np.random.permutation(max_iter)
#     models_array = np.array(models)
#     models_perm = models_array[perm]
#     X_perm = X[perm,:]
#     y_perm = y[perm,:]
#     return  X_perm[:n_train, :], X_perm[n_train:,:], y_perm[:n_train,:], y_perm[n_train:,:], models_perm[:n_train], models_perm[n_train:]

# X_train, X_test, y_train, y_test, models_train, models_test = train_test_split2(X, y, models, test_size=0.1)

# #from sklearn.model_selection import train_test_split
# #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# reg = ExtraTreesRegressor()
# #reg = RandomForestRegressor()
# #reg = DecisionTreeRegressor() #KNeighborsRegressor() #DecisionTreeRegressor() #RandomForestRegressor() #ExtraTreesRegressor(),RadiusNeighborsRegressor()
# #reg = KNeighborsRegressor(n_neighbors=5) 
# multi_reg = MultiOutputRegressor(reg).fit(X_train, y_train)
# yhat = multi_reg.predict(X_test)

# #xg-boost 
# #reg = xgb.XGBRegressor(tree_method="hist", multi_strategy="multi_output_tree") #メモリが膨大
# # reg = xgb.XGBRegressor(tree_method="hist")
# # reg.fit(X_train, y_train)
# # yhat = multi_reg.predict(X_test)

# r2_score(y_test, yhat)
```

```{.python.marimo}
# name_2 = 'tai_j100_m100_1.data'
# with open(f'data({name_2}).pkl', 'rb') as f:
#     compressed_model = f.read()
# serialized_model = zlib.decompress(compressed_model)
# X_1, y_1, multi_reg_1, models_1 = pickle.loads(serialized_model)
```

```{.python.marimo}
# from scipy.stats.mstats import gmean
# yhat_1 = multi_reg_1.predict(X_test)
# name_3 = 'tai_j100_m100_1.data'
# n_1 = 100
# m_1 = 100
# ratio = []
# for iter_1 in range(len(X_test)):
#     opt_val, slow_machine_1, increase_1 = models_test[iter_1]
#     model_5, xx_2 = make_benchmark_model(name_3, time_limit=30, slow_machine=int(slow_machine_1), increase=increase_1)
#     st = []
#     for i_4 in range(n_1):
#         for j in range(m_1):
#             st.append((yhat_1[iter_1][j + m_1 * i_4], model_5.act[j + m_1 * i_4].name + ' ' + model_5.act[j + m_1 * i_4].modes[0].name))
#     st.sort()
#     f_1 = open('optseq_best_act.txt', 'w')
#     f_1.write('source ---\n')
#     for line in st:
#         f_1.write(line[1] + '\n')
#     f_1.write('sink ---\n')
#     f_1.close()
#     model_5.Params.MaxIteration = 1
#     model_5.Params.OutputFlag = 0
#     model_5.Params.Initial = True
#     model_5.optimize(init_fn='optseq_best_act.txt')
#     ratio.append(model_5.ObjVal / opt_val)
#     print(iter_1, model_5.ObjVal, opt_val, model_5.ObjVal / opt_val)
# print(gmean(ratio))
```

```{.python.marimo}
#| hide
# #エンコード・デコード法の実験 (Fast Approximations for Job Shop Scheduling:
# # A Lagrangian Dual Deep Learning Method と類似のデータを生成)

# name = "tai_j10_m10_1.data"
# #name ="tai_j100_m10_1.data"
# #name ="tai_j1000_m10_1.data"
# #name ="tai_j100_m100_1.data"
# #name = "tai_j1000_m100_1.data"
# # name ="tai_j1000_m1000_1.data"

# prec_dic = defaultdict(int)
# equality_dic = defaultdict(int)

# m = 10
# num_steps = 10
# #機械学習用
# max_iter = m*num_steps
# X = np.zeros((max_iter,2*m*m), dtype=int) #機械の番号と作業時間を特徴とする
# y = np.zeros((max_iter,m*m), dtype=int) #開始時刻を予測する

# iter = 0
# for slow_machine in range(m):
#     print("\n m=", slow_machine)
#     for increase in np.linspace(1.0, 1.5, num_steps):
#         print(increase, end="")
#         model, xx = make_benchmark_model(name, 10000, slow_machine=slow_machine, increase=increase)
#         model.Params.TimeLimit = 10

#         horizon = 1000000
#         #print("horizon=", horizon)
#         model.Params.OutputFlag = False

#         # starting the monitoring
#         #tracemalloc.start()

#         start_time = time.time()
#         cpmodel = transform_to_cp(model, horizon)
#         end_time = time.time()

#         # print(end_time - start_time)
#         #print(model.ObjVal, model.Status)

#         n = len(model.act)
#         #以下は O(n^2)かかる
#         # equality_matrix = np.zeros( (n,n) )
#         # prec_matrix = np.zeros( (n,n) )
#         # st, fi = [], []
#         # for a in model.act:
#         #     st.append( a.start )
#         #     fi.append( a.completion )
#         # for i in range(n-1):
#         #     for j in range(i+1,n):
#         #         if fi[i]==st[j]:
#         #             equality_matrix[i,j] = 1
#         #         if st[i] < st[j]:
#         #             prec_matrix[i,j] = 1
#         #         else:
#         #             prec_matrix[j,i] = 1                    
#         # key = tuple(map(tuple, equality_matrix))
#         # equality_dic[key].append(model.ObjVal)

#         # key = tuple(map(tuple, prec_matrix))
#         # prec_dic[key].append(model.ObjVal)
#         #半順序を保管 =>開始時刻のみ保管
#         st, fi = [], []
#         for i, a in enumerate(model.act):
#             st.append( (a.start, i) )
#             fi.append( (a.completion, i) )
#         st.sort()
#         layers = [] #半順序を表すリスト
#         now = -1 
#         for (st, job_id) in st:
#             if st > now:
#                 now = st
#                 layer = [job_id]
#                 layers.append(layer)
#             elif st == now: #同じ層
#                 layer.append(job_id)
#             else:
#                 print("wrong order")
#         for k, layer in enumerate(layers[:-1]):
#             for i in layer:
#                 for j in layers[k+1]:
#                     prec_dic[i,j] += 1
#         #終了・開始がつながっているジョブ対を保管
#         start_jobs = defaultdict(list) #時刻に開始しているジョブの集合
#         connected_jobs = [] #終了・開始がつながっているジョブ対のリスト
#         for i, a in enumerate(model.act):
#             start_jobs[a.start].append(i) 
#         for i, a in enumerate(model.act):
#             fi = a.completion
#             if fi in start_jobs: #終了時刻に開始するジョブのリストがある
#                 for j in  start_jobs[fi]:
#                     connected_jobs.append( (i,j) )
#         for (i,j) in connected_jobs:
#             equality_dic[i,j] += 1    
#         iter+=1
```

```{.python.marimo}
#| hide
# key = tuple(map(tuple, prec_matrix))
# dic = {key:model.ObjVal}
# 
# equality_matrix = np.zeros( (n,n) )
# prec_matrix = np.zeros( (n,n) )
# for key in prec_dic:
#     #print(len(prec_dic[key]))
#     array = np.array(key)
#     prec_matrix += array*len(prec_dic[key])

# for key in equality_dic:
#     #print(len(equality_dic[key]))
#     array = np.array(key)
#     equality_matrix += array*len(equality_dic[key])
```

```{.python.marimo}
#import matplotlib.pyplot as plt
#prec_dicの枝をもつグラフ上で最大重みの半順序（acyclic subgraph）を求める？

# prec_matrix.shape = (-1)
# prec_matrix.sort()
# #prec_matrix[-3000:]
# plt.hist(prec_matrix[:], bins=20, alpha=0.7, color='blue', edgecolor='black');
#plt.hist(prec_dic.values(), bins=20, alpha=0.7, color='blue', edgecolor='black');
```

```{.python.marimo}
#equality_matrix.shape = (-1)
# equality_matrix.sort()
# #equality_matrix[-100:]
# #hist, bins = np.histogram(equality_matrix)
# plt.hist(equality_matrix[-100:], bins=20, alpha=0.7, color='blue', edgecolor='black');
#plt.hist(equality_dic.values(), bins=20, alpha=0.7, color='blue', edgecolor='black');
```

```{.python.marimo}
# import pickle
# import zlib
# name = "tai_j1000_m100_1.data"

# # 圧縮されたモデルをファイルから読み込み
# with open(f"model({name}).pkl", "rb") as f:
#     compressed_model = f.read()
# # 解凍
# serialized_model = zlib.decompress(compressed_model)
# # モデルを復元
# model = pickle.loads(serialized_model)

# # モデルを使用
# #print(model)
```

## データ可視化関数 data_visualize

作業時間，納期などを可視化する．

```{.python.marimo}
#| export
def data_visualize(model: Model):
    model.update()
    scale = 10
    duration = []
    duedate = []
    for a_name, a in model.activities.items():
        duedate.append(a.duedate)
        for m in a.modes:
            duration.append(m.duration)

    df = pd.DataFrame({"duration": duration})
    df2 = pd.DataFrame({"duedate": duedate})
    fig = px.histogram(df, x="duration") 
    fig2 = px.histogram(df2, x="duedate")  
    return fig, fig2
```

```{.python.marimo}
# name_4 = 'tai_j10_m10_1.data'
# model_6 = make_benchmark_model(name_4, 10000)
# fig_3, fig2 = data_visualize(model_6)
# fig_3
```

<!-- ## Shifting Bottleneck法

大規模なジョブショップスケジューリング問題に対する解法としてShifting Bottleneck法を準備する．

1台の機械に対するスケジューリング問題をCPソルバーで解く操作を繰り返すことによって， 複数機械のスケジューリング問題の近似解を算出する．

クラス

- Job: 機械の番号 r と作業時間 p　を引数としたジョブ（正確にはオペレーション）のクラス
- Jobshop: NetworkXの有向グラフクラスから派生させたクラスであり， ジョブをノードとして追加することによって， ジョブショップスケジューリング問題のネットワークを定義する，
- Shift: Jobshopから派生させたShifting Bottleneck法のメインクラス．

上のクラスを用いてShifting Bottleneck法を実行する関数 shifting_bottleneckを構築する． -->

```{.python.marimo}
class Job(object):
    """
    A class that creates jobs.

    Parameters
    ----------
    r: list - 機械の順序を表すリスト
    p: list - 作業時間のリスト
    """

    def __init__(self, Id, r, p):
        self.Id = Id
        self.r = r
        self.p = p

class Jobshop(nx.DiGraph):
    """
    A class that creates a directed graph of a jobshop.

    We formulate the tasks of the jobshop as nodes in a directed graph, add the processing 
    times of the tasks as attributes to the task nodes. A flag "dirty" was added so when 
    some topological changes are carried the method "_update" is called first to update 
    the makespan and critical path values. Once the update is finished, the updated 
    makespan is returned.

    Methods
    -------
    handleJobRoutings(jobs)
        Creates the edges of the graph that represents the given route and also adds 
        the origin and finishing nodes.

    handleJobProcessingTimes(jobs)
        Creates the nodes of the graph that represent the tasks of a job.

    makeMachineSubgraphs()
        For every given machine creates a subgraph.

    addJobs(jobs)
        Handles the routine to add a jobs to the graph and the subgraphs.

    output()
        Prints the output. 

    _forward

    _backward

    _computeCriticalPath

    _update

    Properties
    ----------
    makespan

    criticalPath

    """

    def __init__(self):
        super().__init__()
        self.machines = {}
        self.add_node('U', p=0)
        self.add_node('V', p=0)
        self._dirty = True
        self._makespan = -1
        self._criticalPath = None

    def add_node(self, *args, **kwargs):
        self._dirty = True
        super().add_node(*args, **kwargs)

    def add_nodes_from(self, *args, **kwargs):
        self._dirty = True
        super().add_nodes_from(*args, **kwargs)

    def add_edge(self, *args):
        self._dirty = True
        super().add_edge(*args)

    def add_edges_from(self, *args, **kwargs):
        self._dirty = True
        super().add_edges_from(*args, **kwargs)

    def remove_node(self, *args, **kwargs):
        self._dirty = True
        super().remove_node(*args, **kwargs)

    def remove_nodes_from(self, *args, **kwargs):
        self._dirty = True
        super().remove_nodes_from(*args, **kwargs)

    def remove_edge(self, *args):
        self._dirty = True
        super().remove_edge(*args)

    def remove_edges_from(self, *args, **kwargs):
        self._dirty = True
        super().remove_edges_from(*args, **kwargs)

    def handleJobRoutings(self, jobs):
        for j in jobs.values():
            self.add_edge('U', (j.r[0], j.Id))
            for m, n in zip(j.r[:-1], j.r[1:]):
                self.add_edge((m, j.Id), (n, j.Id))
            self.add_edge((j.r[-1], j.Id), 'V')

    def handleJobProcessingTimes(self, jobs):
        for j in jobs.values():
            for m, p in zip(j.r, j.p):
                self.add_node((m, j.Id), p=p)

    def makeMachineSubgraphs(self):
        machineIds = set((ij[0] for ij in self if ij[0] not in ('U', 'V')))
        for m in machineIds:
            self.machines[m] = self.subgraph((ij for ij in self if ij[0] == m not in ('U', 'V')))

    def addJobs(self, jobs):
        self.handleJobRoutings(jobs)
        self.handleJobProcessingTimes(jobs)
        self.makeMachineSubgraphs()

    def output(self):
        for m in sorted(self.machines):
            for j in sorted(self.machines[m]):
                print('{}: {}'.format(j, self.node[j]['C']))

    def _forward(self):
        for n in nx.topological_sort(self):
            S = max([self.nodes[j]['C'] for j in self.predecessors(n)], default=0)
            self.add_node(n, S=S, C=S + self.nodes[n]['p'])

    def _backward(self):
        for n in list(reversed(list(nx.topological_sort(self)))):
            Cp = min([self.nodes[j]['Sp'] for j in self.successors(n)], default=self._makespan)
            self.add_node(n, Sp=Cp - self.nodes[n]['p'], Cp=Cp)

    def _computeCriticalPath(self):
        G = set()
        for n in self:
            if self.nodes[n]['C'] == self.nodes[n]['Cp']:
                G.add(n)
        self._criticalPath = self.subgraph(G)

    @property
    def makespan(self):
        if self._dirty:
            self._update()
        return self._makespan

    @property
    def criticalPath(self):
        if self._dirty:
            self._update()
        return self._criticalPath

    def _update(self):
        self._forward()
        self._makespan = max(nx.get_node_attributes(self, 'C').values())
        self._backward()
        self._computeCriticalPath()
        self._dirty = False

class Shift(Jobshop):

    def output(self):
        print('makespan: ', self.makespan)
        for i in self.machines:
            print('Machine: ' + str(i))
            s = '{0:<7s}'.format('jobs:')
            for ij in sorted(self.machines[i]):
                if ij in ('U', 'V'):
                    continue
                s = s + '{0:>5d}'.format(ij[1])
            print(s)
            s = '{0:<7s}'.format('p:')
            for ij in sorted(self.machines[i]):
                if ij in ('U', 'V'):
                    continue
                s = s + '{0:>5d}'.format(self.nodes[ij]['p'])
            print(s)
            s = '{0:<7s}'.format('r:')
            for ij in sorted(self.machines[i]):
                if ij in ('U', 'V'):
                    continue
                s = s + '{0:>5d}'.format(self.nodes[ij]['S'])
            print(s)
            s = '{0:<7s}'.format('d:')
            for ij in sorted(self.machines[i]):
                if ij in ('U', 'V'):
                    continue
                s = s + '{0:>5d}'.format(self.nodes[ij]['Cp'])
            print(s)
            print('\n')

    def computeLmax2(self, remain: list, horizon: int, precedence: List=[]):
        result = []
        for m in remain:
            cpmodel = CpModel()
            jobs = list(self.machines[m])
            start, end, act, late = ({}, {}, {}, {})
            activities_on_machine = []
            all_lates = []
            for j in jobs:
                start[j] = cpmodel.new_int_var(self.nodes[j]['S'], horizon, f'start({j})')
                end[j] = cpmodel.new_int_var(self.nodes[j]['S'] + self.nodes[j]['p'], horizon, f'end({j})')
                late[j] = cpmodel.new_int_var(0, horizon - self.nodes[j]['Cp'], f'late({j})')
                all_lates.append(late[j])
                act[j] = cpmodel.new_interval_var(start[j], self.nodes[j]['p'], end[j], f'act({j})')
                activities_on_machine.append(act[j])
                cpmodel.add(late[j] >= end[j] - self.nodes[j]['Cp'])
            cpmodel.add_no_overlap(activities_on_machine)
            obj = cpmodel.new_int_var(0, horizon, 'max_lateness')
            cpmodel.add_max_equality(obj, all_lates)
            cpmodel.minimize(obj)
            if len(precedence) > 0:
                for st, fi, delay in precedence:
                    cpmodel.add(start[st] + delay <= start[fi])
            solver = CpSolver()
            solver.parameters.max_time_in_seconds = 1000
            solver.parameters.log_search_progress = False
            status = solver.Solve(cpmodel)
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                seq = [(solver.value(start[j]), j) for j in jobs]
                seq.sort()
                s = [j for _, j in seq]
                result.append((m, int(solver.ObjectiveValue()), s))
            else:
                print('Solver stopped with status', status)
        return result

def shifting_bottleneck(js: Shift, horizon: int=1000000, LOG: bool=False):
    remain = set(js.machines.keys())
    count = 0
    while len(remain) > 0:
        js.criticalPath
        result = js.computeLmax2(remain, horizon)
        max_lmax = -1
        max_machine = -1
        best_seq = None
        for idx, lmax, seq in result:
            if lmax > max_lmax:
                max_lmax = lmax
                max_machine = idx
                best_seq = seq
        print(max_lmax, max_machine)
        if count == 1:
            best_seq = [(2, 3), (2, 2), (2, 1)]
        pred = best_seq[0]
        edges = []
        for succ in best_seq[1:]:
            edges.append((pred, succ))
            pred = succ
        while True:
            G = js.copy()
            G.add_edges_from(edges)
            if nx.is_directed_acyclic_graph(G) == True:
                break
            S = set(js.machines[max_machine])
            G.remove_nodes_from(['U', 'V'])
            remove_zero_indegree_nodes(G)
            remove_zero_outdegree_nodes(G)
            label, prev = ({}, {})
            precedence = []
            for i in S:
                if i not in G:
                    continue
                label[i] = 0
                for k, j in nx.dfs_edges(G, source=i):
                    label[j] = label[k] + G.nodes[k]['p']
                    prev[j] = k
                    if i in G[j] and j in S:
                        print('found a short cycle', i, j, label[j])
                        precedence.append((i, j, label[j]))
            if len(precedence) > 0:
                result = js.computeLmax2([max_machine], horizon, precedence)
                idx, lmax, best_seq = result[0]
                pred = best_seq[0]
                edges = []
                for succ in best_seq[1:]:
                    edges.append((pred, succ))
                    pred = succ
                break
        count = count + 1
        js.add_edges_from(edges)
        remain.remove(max_machine)
```

<!-- ### shifting _bottleneckの使用例 -->

```{.python.marimo}
# magic command not supported in marimo; please file an issue to add support
# %time
#| # hide
# # js = Shift()
# 
# # jobs = {}
# # jobs[1] = Job(1, [1,2,3], [10, 8, 4])
# # jobs[2] = Job(2, [2,1,4,3], [8,3,5,6])
# # jobs[3] = Job(3, [1,2,4], [4,7,3])
# 
# # js.addJobs(jobs)
# # shifting_bottleneck(js, horizon=100, LOG=False)
# # js.output()
# #ベンチマークの実験
# #name = "ft06.txt"
# name = "tai_j10_m10_1.data"
# #name ="tai_j100_m10_1.data"
# #name ="tai_j1000_m10_1.data"
# #name ="tai_j100_m100_1.data"
# #name = "tai_j1000_m100_1.data"
# #name ="tai_j1000_m1000_1.data"
# 
# fname = f"./data/optseq/{name}"
# 
# horizon = 1000000
# f = open(fname, "r")
# lines = f.readlines()
# f.close()
# n, m = map(int, lines[0].split())
# print("n,m=", n, m)
# 
# js = Shift()
# 
# jobs ={}
# for i in range(n):
#     L = list(map(int, lines[i + 1].split()))
#     machines, durations = [], []
#     for j in range(m):
#         machines.append( L[2 * j] + 1 )
#         durations.append( L[2 * j + 1] )
#     jobs[i+1] = Job(i+1, machines, durations)
#     
# js.addJobs(jobs)
# 
# remain = set(js.machines.keys()) #残りの機械
# count = 0
# while len(remain) > 0:
#     js.criticalPath
#     #if LOG: js.output()
#     
#     result = js.computeLmax2(remain, horizon) #use cp-sat
#     #print("result=", result) #machine, obj, seq
#     max_lmax = -1
#     max_machine = -1
#     best_seq = None
#     for idx, lmax, seq in result:
#         if lmax > max_lmax:
#             max_lmax = lmax
#             max_machine = idx
#             best_seq = seq
#     print(count, max_machine, max_lmax)
# 
#     # if count ==1:
#     #     best_seq =[(2, 3), (2, 2), (2, 1)]
# 
#     #閉路がある場合には切除平面を追加
#     pred = best_seq[0]
#     edges = []
#     for succ in best_seq[1:]:
#         edges.append( (pred, succ) )
#         pred = succ
#     #print("edges=", edges)   
#     while True:
#         G = js.copy()
#         G.add_edges_from(edges)
#         if nx.is_directed_acyclic_graph(G)==True: #閉路がない
#             break
#             
#         #閉路があるかどうかを確認し，あるならカットを追加して再求解
#         S = set(js.machines[max_machine])
#         G.remove_nodes_from(['U', 'V'])
#         #print("before preprocessing", len(G.edges()))
#         remove_zero_indegree_nodes(G)
#         remove_zero_outdegree_nodes(G)
#     
#         label, prev = {}, {}
#         precedence =[]
#         #find shortest paths between 2 jobs on the machine
#         for i in S:
#             if i not in G:
#                 continue
#             #print("search from", i)
#             label[i] = 0
#             for (k,j) in nx.bfs_edges(G, source=i):
#                 label[j] = label[k] + G.nodes[k]['p']
#                 prev[j] = k
#                 if i in G[j] and j in S:
#                     print("found a cycle", i,j, label[j])
#                     precedence.append( (i,j,label[j]) )
#                     break #only one cycle
#                     
#         if len(precedence)>0:
#             #カットの追加して再求解
#             result = js.computeLmax2([max_machine], horizon, precedence) #use cp-sat
#             idx, lmax, best_seq = result[0]
#             #print("best_seq=", best_seq)
#             pred = best_seq[0]
#             edges = []
#             for succ in best_seq[1:]:
#                 edges.append( (pred, succ) )
#                 pred = succ
#             #print("edges=", edges)
#             
#     # js.add_edges_from(edges) #機械での順序を追加
#     count +=1
#     # if count ==2:
#     #     break
#     js.add_edges_from(edges) #
#     remain.remove(max_machine)
```

```{.python.marimo unparsable="true"}
#| hide

# SB法の課題: 
# 1. 遅れを伴う先行制約を付加しないと，稀に実行不能になる（閉路ができる） => 
#    閉路ができたらそれを禁止する先行制約を追加！！！ or 高速に先行制約を追加する方法（閉路の数が指数オーダーで増大するので，大規模問題例では使えない）
# 2. 機械学習でLmaxを推定して高速化

# m = 2
# jobs = set(js.machines[m])
# #各ジョブから他のジョブへの最長路を求める
# order = [] #機械m上でのトポロジカルソート
# label = defaultdict(int)
# for n in nx.topological_sort(js):
#     if n in jobs:
#         order.append(n)
# for i in orde


r[:-1]:
#     print("search from", i)
#     label[i] = 0
#     for (k,j) in nx.dfs_edges(js, source=i):
#         label[j] = label[k] + js.nodes[k]['p']
#         print(k,j, label[j])
#         if j in jobs:
#             print("found a path", j, label[j])
```

```{.python.marimo}
#| hide
# 機械上のジョブ（点）を１つの点にshrinkさせてから，networkXで単純閉路を求める
# 他にも， ダミーの始点，終点を削除した上で，入次数，出次数が0の点を削除したり，　機械上のジョブ（点）から直接つながっていない有向枝は縮約しておくなどの前処理が考えられる
def shrink_subset_to_node(G, S):
    """
    Shrink a subset of nodes S in graph G to a single node.

    Parameters:
        G (nx.DiGraph): Directed graph.
        S (list): List of nodes to be shrunk.

    Returns:
        None (in-place modification of G).
    """
    # Collect all edges incident to the subset S
    succ_edges, pred_edges = [], []
    for u in S:
        for v in G.successors(u):
            if v not in S:
                succ_edges.append((u, v))
        for v in G.predecessors(u):
            if v not in S:
                pred_edges.append((v, u))
    G.remove_nodes_from(S)
    new_node = tuple(S)  # Create a tuple to represent the subset as a single node
    G.add_node(new_node)
    for (u,v) in succ_edges:
        G.add_edge(new_node, v)
    for (v,u) in pred_edges:
        G.add_edge(v,new_node)   
    return succ_edges, pred_edges 

# Example usage:
G = nx.DiGraph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])

# Shrink subset {2, 3, 4} to a single node
succ_edges, pred_edges = shrink_subset_to_node(G, [2, 3, 4])

# Display the modified graph
print("Nodes in the modified graph:", G.nodes())
print("Edges in the modified graph:", G.edges())
print(succ_edges, pred_edges)
```

```{.python.marimo}
from collections import deque

# def remove_zero_indegree_nodes(G):
#     """
#     Remove nodes from graph G whose in-degree is 0 and update adjacent nodes' degrees.

#     Parameters:
#         G (nx.DiGraph): Acyclic directed graph.

#     Returns:
#         None (in-place modification of G).
#     """
#     degree = {node: G.in_degree(node) for node in G.nodes()}
#     zero_indegree_nodes = deque()
#     for node in degree:
#         if degree[node] == 0:
#             zero_indegree_nodes.append(node)
#     while True:
#         if len(zero_indegree_nodes) == 0:
#             break
#         node = zero_indegree_nodes.popleft()
#         successors = list(G.successors(node))
#         G.remove_node(node)
#         for succ in successors:
#             degree[succ] = degree[succ] - 1
#             if degree[succ] == 0:
#                 zero_indegree_nodes.append(succ)

# def remove_zero_outdegree_nodes(G):
#     degree = {node: G.out_degree(node) for node in G.nodes()}
#     zero_outdegree_nodes = deque()
#     for node in degree:
#         if degree[node] == 0:
#             zero_outdegree_nodes.append(node)
#     while True:
#         if len(zero_outdegree_nodes) == 0:
#             break
#         node = zero_outdegree_nodes.popleft()
#         predecessors = list(G.predecessors(node))
#         G.remove_node(node)
#         for pred in predecessors:
#             degree[pred] = degree[pred] - 1
#             if degree[pred] == 0:
#                 zero_outdegree_nodes.append(pred)
# G_1 = nx.DiGraph()
# G_1.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4), (4, 3), (3, 5), (4, 5), (4, 6)])
# remove_zero_indegree_nodes(G_1)
# remove_zero_outdegree_nodes(G_1)
# print('Nodes in the modified graph:', G_1.nodes())
# print('Edges in the modified graph:', G_1.edges())
```

```{.python.marimo}
# m_2 = 2
# S = set(js.machines[m_2])
# G_2 = js.copy()
# G_2.remove_nodes_from(['U', 'V'])
# shrink_subset_to_node(G_2, S)
# print('before preprocessing', len(G_2.edges()))
# remove_zero_indegree_nodes(G_2)
# remove_zero_outdegree_nodes(G_2)
# print('after preprocessing', len(G_2.edges()))
# G_2.edges()
# for p in nx.simple_cycles(G_2):
#     print(p)
```

```{.python.marimo}
# nx.is_directed_acyclic_graph(js._criticalPath)
```

```{.python.marimo}
#| hide

# shifting_bottleneck(js, horizon=1000000, LOG=False)

# #js.output()
# print("makespan=", js.makespan)
```

```{.python.marimo}
def find_delayed_precedence(p: List, js: nx.DiGraph, S, succ_edges, pred_edges):
    idx = p.index(tuple(S))

    def next_idx(p, idx):
        if len(p) - 1 == idx:
            return 0
        else:
            return idx + 1
    for u, v in succ_edges:
        if v == p[next_idx(p, idx)]:
            head_job = u
            break
    for u, v in pred_edges:
        if u == p[idx - 1]:
            tail_job = v
            break
    delay = js.nodes[head_job]['p']
    i = next_idx(p, idx)
    while i != idx:
        job = p[i]
        delay = delay + js.nodes[job]['p']
        i = next_idx(p, i)
    return (head_job, tail_job, delay)
```

```{.python.marimo}
#| hide
# shifting_bottleneck(js, horizon=100000, LOG=False)

# m = 2
# S = set(js.machines[m])
# G = js.copy()
# G.remove_nodes_from(['U', 'V'])
# succ_edges, pred_edges = shrink_subset_to_node(G, S)
# print("before preprocessing", len(G.edges()))
# remove_zero_indegree_nodes(G)
# remove_zero_outdegree_nodes(G)
# print("after preprocessing",len(G.edges()))
# # networkX の simple_cycles(G, length_bound=None)[source] の利用
# precedence = []
# for p in nx.simple_cycles(G):
#     #print("find a circuit=", p)
#     head_job, tail_job, delay = find_delayed_precedence(p, js)
#     precedence.append((head_job, tail_job, delay))
#     print(head_job, tail_job, delay)
```

```{.python.marimo}
#| hide

# test or-tools cp-sat
#name = "tai_j10_m10_1.data"
# name ="tai_j100_m10_1.data"
# name ="tai_j1000_m10_1.data"
# name ="tai_j100_m100_1.data"
# name = "tai_j1000_m100_1.data"
# name ="tai_j1000_m1000_1.data"

# fname = f"./data/optseq/{name}"

# f = open(fname, "r")
# lines = f.readlines()
# f.close()
# n, m = map(int, lines[0].split())
# print("n,m=", n, m)

# # prepare data
# machine, proc_time = {}, {}
# for i in range(n):
#     L = list(map(int, lines[i + 1].split()))
#     for j in range(m):
#         machine[i, j] = L[2 * j]
#         proc_time[i, j] = L[2 * j + 1]
# jobs_data = []
# for i in range(n):
#     row = []
#     for j in range(m):
#         row.append((machine[i, j], proc_time[i, j]))
#     jobs_data.append(row)

# import collections
# from ortools.sat.python import cp_model

# model = cp_model.CpModel()

# machines_count = 1 + max(task[0] for job in jobs_data for task in job)
# all_machines = range(machines_count)

# # Computes horizon dynamically as the sum of all durations.
# horizon = sum(task[1] for job in jobs_data for task in job)

# # Named tuple to store information about created variables
# task_type = collections.namedtuple("task_type", "start end interval")
# # Named tuple to manipulate solution information
# assigned_task_type = collections.namedtuple(
#     "assigned_task_type", "start job index duration"
# )

# # Creates job intervals and add to the corresponding machine lists
# all_tasks = {}
# machine_to_intervals = collections.defaultdict(list)

# for job_id, job in enumerate(jobs_data):
#     for task_id, task in enumerate(job):
#         machine = task[0]
#         duration = task[1]
#         suffix = "_%i_%i" % (job_id, task_id)
#         start_var = model.NewIntVar(0, horizon, "start" + suffix)
#         end_var = model.NewIntVar(0, horizon, "end" + suffix)
#         interval_var = model.NewIntervalVar(
#             start_var, duration, end_var, "interval" + suffix
#         )
#         all_tasks[job_id, task_id] = task_type(
#             start=start_var, end=end_var, interval=interval_var
#         )
#         machine_to_intervals[machine].append(interval_var)

# # Create and add disjunctive constraints
# for machine in all_machines:
#     model.AddNoOverlap(machine_to_intervals[machine])

# # Precedences inside a job
# for job_id, job in enumerate(jobs_data):
#     for task_id in range(len(job) - 1):
#         model.Add(
#             all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end
#         )

# # Makespan objective
# obj_var = model.NewIntVar(0, horizon, "makespan")
# model.AddMaxEquality(
#     obj_var,
#     [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)],
# )
# model.Minimize(obj_var)

# # Solve model
# solver = cp_model.CpSolver()
# solver.parameters.max_time_in_seconds = 360.0
# status = solver.Solve(model)


# # Create one list of assigned tasks per machine.
# assigned_jobs = collections.defaultdict(list)
# for job_id, job in enumerate(jobs_data):
#     for task_id, task in enumerate(job):
#         machine = task[0]
#         assigned_jobs[machine].append(
#             assigned_task_type(
#                 start=solver.Value(all_tasks[job_id, task_id].start),
#                 job=job_id,
#                 index=task_id,
#                 duration=task[1],
#             )
#         )

# # Create per machine output lines.
# output = ""
# for machine in all_machines:
#     # Sort by starting time.
#     assigned_jobs[machine].sort()
#     sol_line_tasks = "Machine " + str(machine) + ": "
#     sol_line = "           "

#     for assigned_task in assigned_jobs[machine]:
#         name = "job_%i_%i" % (assigned_task.job, assigned_task.index)
#         # Add spaces to output to align columns.
#         sol_line_tasks += "%-10s" % name

#         start = assigned_task.start
#         duration = assigned_task.duration
#         sol_tmp = "[%i,%i]" % (start, start + duration)
#         # Add spaces to output to align columns.
#         sol_line += "%-10s" % sol_tmp

#     sol_line += "\n"
#     sol_line_tasks += "\n"
#     output += sol_line_tasks
#     output += sol_line

# # Finally print the solution found.
# print("Optimal Schedule Length: %i" % solver.ObjectiveValue())
# print(output)
```

## Notionのガントチャート用のデータフレームを生成する関数 make_gantt_for_notion

Notion のtimelineを用いてガントチャートを表示する．

使用法：

- 生成したデータフレームをcsvファイルとして保存
- https://www.notion.so/ でimportしてからタイムラインページを生成して表示
- もしくは既存のタイムラインページに merge with csv でインポート

引数：

- model : モデルオブジェクト
- start : 開始時刻を表す（pandasの日付時刻型に変換可能な文字列）文字列．形式はpandasのto_datetimeに認識される文字列で，例えば"2024/1/1 00:00:00"．（既定値）．
- period : 時間の単位を表す文字列． "days"（日）， "seconds"（秒），　"minutes"（分）， "hours（時間）の何れか． 既定値は "days"

返値：

- df : Notionのタイムライン形式のデータフレーム

```{.python.marimo}
#| export
def make_gantt_for_notion(model, start="2020/1/1 00:00:00", period="days"):
    """
    notionのガントチャートを生成する関数
    """
    start = pd.to_datetime(start)

    def time_convert_long(periods):
        if period =="days":
            time_ = start + dt.timedelta(days=float(periods))
        elif period == "hours":
            time_ = start + dt.timedelta(hours=float(periods))
        elif period == "minutes":
            time_ = start + dt.timedelta(minutes=float(periods))
        elif period == "seconds":
            time_ = start + dt.timedelta(seconds=float(periods))
        else:
            raise TypeError("pariod must be 'days' or 'seconds' or minutes' or 'days'")
        return time_.strftime("%Y年%m月%d日 %H:%M" ) #"%B %d, %Y %I:%M %p")

    L = []
    Name,Assign,Blocked_by,Blocking,Date,Property,Property_1,Status = [],[],[],[],[],[],[],[]
    #Name, Assign, Date, Property,Property_1,Status =[],[],[],[],[],[]
    for i in model.activities:
        a = model.activities[i]
        st = time_convert_long(a.start)
        fi = time_convert_long(a.completion-1)  #Notionのガントチャートは終了時刻を含むので， １単位時間前まで使用するように変更

        Name.append(a.name)

        Assign.append( a.selected.name )     #選択されたモード
        Date.append( str(st)+" → "+ str(fi)) #開始・終了
        Property.append( f"duedate: {a.duedate}  weight:{a.weight}" )
        Property_1.append(f"execute: {a.execute}" )
        Blocked_by.append(" ")
        Blocking.append(" ")
        Status.append( "Not started"  ) 


    df = pd.DataFrame( {"Name":Name, "Assign":Assign, "Blocked by":Blocked_by, "Blocking": Blocking, 
                        "日付":Date, "Property": Property, "Property 1": Property_1, "Status": Status} )
    return df
```

### make_gantt_for_notion関数の使用例

```{.python.marimo}
_start = dt.datetime(2024, 1, 1, 0, 0)
_start = _start.strftime('%Y-%m-%d %H:%M:%S')
_df = make_gantt_for_notion(model, _start, period='minutes')
#_df.to_csv('gantt.csv', index=False)
#_df.head()
```

## Excelのガントチャートを生成する関数 make_gantt_for_excel

```{.python.marimo}
def make_gantt_for_excel(model: Model, start: str='2020/1/1 00:00:00'):
    """
    Excelのガントチャートを生成する関数
    """

    def prepare_res_idx():
        count = 0
        name_list = []
        for res in model.res:
            if len(res.terms) == 0:
                count = count + 1
                name_list.append(res.name)
        n_res = len(name_list)
        if n_res > 0:
            name_dic = {}
            for i, name in enumerate(name_list):
                name_dic[name] = i + 1
            res_idx_list = []
            for idx, i in enumerate(model.activities):
                a = model.activities[i]
                try:
                    res_name = list(a.modes[0].requirement.keys())[0][0]
                    res_idx = name_dic[res_name]
                except IndexError:
                    res_idx = n_res + 1
                res_idx_list.append(res_idx)
        return (res_idx_list, name_list)

    def time_convert_long(periods, period='days'):
        if period == 'days':
            time_ = start + dt.timedelta(days=float(periods))
        elif period == 'hours':
            time_ = start + dt.timedelta(hours=float(periods))
        elif period == 'minutes':
            time_ = start + dt.timedelta(minutes=float(periods))
        elif period == 'seconds':
            time_ = start + dt.timedelta(seconds=float(periods))
        else:
            raise TypeError("pariod must be 'days' or 'seconds' or minutes' or 'days'")
        return time_.strftime('%Y/%m/%d')
    start = pd.to_datetime(start)
    n_job = len(model.activities)
    wb = Workbook()
    ws = wb.active
    max_time = 0
    for i in model.activities:
        a = model.activities[i]
        max_time = max(max_time, int(a.completion))
    ws.append(['基準日', start, '  ', ' ', '    ', '    '])
    for j in range(max_time):
        cell = ws.cell(1, j + 7)
        col = cell.column_letter
        cell.value = f'=LEFT(TEXT({col}2,"aaaa"),1)'
    ws.append(['名称', '開始', '終了', '日数', '納期', '遅れ重み'])
    ws['G2'].value = '=B1'
    for j in range(max_time - 1):
        cell = ws.cell(2, j + 7)
        col = cell.column_letter
        new_cell = ws.cell(2, j + 8)
        new_cell.value = f'={col}2+1'
    for idx, i in enumerate(model.activities):
        L = []
        a = model.activities[i]
        st = time_convert_long(a.start)
        fi = time_convert_long(a.completion)
        if a.duedate != 'inf':
            due = time_convert_long(a.duedate)
        else:
            due = None
        L = [a.name, pd.to_datetime(st), pd.to_datetime(fi), a.selected.duration, due, a.weight]
        ws.append(L)
    for idx, i in enumerate(model.activities):
        a = model.activities[i]
        if a.duedate != 'inf':
            side = Side(style='thick', color='000000')
            ws.cell(idx + 3, 6 + int(a.duedate)).border = Border(right=side)
    ws.cell(1, 2).number_format = 'mm-dd-yy'
    for j in range(max_time):
        ws.cell(2, 7 + j).number_format = 'mm-dd-yy'
    for i in range(n_job):
        ws.cell(i + 3, 2).number_format = 'mm-dd-yy'
        ws.cell(i + 3, 3).number_format = 'mm-dd-yy'
    res_idx_list, name_list = prepare_res_idx()
    ws.append(['', '', '', '', '資源ID', '資源名'] + list(range(max_time)))
    for j, res in enumerate(name_list):
        ws.append(['', '', '', '', j + 1, res])
    for j, res in enumerate(name_list):
        for t in range(max_time):
            cell = ws.cell(n_job + 4 + j, 7 + t)
            col = cell.column_letter
            cell.value = f'=COUNTIF({col}3:{col}{n_job + 2},{j + 1})'
    for i in range(n_job):
        for j in range(max_time):
            cell = ws.cell(i + 3, j + 7)
            col = cell.column_letter
            cell.value = f'=IF(AND({col}2<C{i + 3},{col}2>=B{i + 3}),{res_idx_list[i]},0)'
    redFill = PatternFill(start_color='0099CC', end_color='0099CC', fill_type='solid')
    cell = ws.cell(3, 6 + max_time)
    col = cell.column_letter
    ws.conditional_formatting.add(f'G3:{col}{2 + n_job}', FormulaRule(formula=['G3>=1'], fill=redFill))
    for j in range(max_time):
        cell = ws.cell(2, j + 7)
        cell.alignment = Alignment(horizontal='center', vertical='center', textRotation=90, wrap_text=False)
    ws.row_dimensions[2].height = 60
    ws.column_dimensions['B'].width = 10
    ws.column_dimensions['C'].width = 10
    ws.column_dimensions['E'].width = 10
    for j in range(max_time):
        cell = ws.cell(1, j + 7)
        col = cell.column_letter
        ws.column_dimensions[col].width = 3
    for j, res in enumerate(name_list):
        c1 = ScatterChart()
        c1.title = f'{j + 1}. 資源 {res}'
        c1.style = 13
        c1.y_axis.title = '資源使用量'
        c1.x_axis.title = '期'
        xvalues = Reference(ws, min_col=7, min_row=n_job + 3, max_col=7 + max_time)
        yvalues1 = Reference(ws, min_col=7, min_row=n_job + 4 + j, max_col=7 + max_time)
        series1 = Series(yvalues1, xvalues, title_from_data=True)
        series1.marker.symbol = 'triangle'
        series1.marker.graphicalProperties.solidFill = 'FF0000'
        series1.marker.graphicalProperties.line.solidFill = 'FF0000'
        series1.graphicalProperties.line.noFill = True
        c1.series.append(series1)
        ws.add_chart(c1, f'A{5 + n_job + len(name_list) + j * 20}')
    return wb
```

### make_gantt_for_excelの使用例

```{.python.marimo}
_wb = make_gantt_for_excel(model, start="2020/5/1 00:00:00")
_wb.save("gantt.xlsx")
```

## プロジェクトスケジューリング用のExcel簡易入力テンプレートを出力する関数  optseq_project_excel

TODO: 順序依存の段取り

注）この関数はSCMOPTに含まれるスケジューリング最適化システム用関数です．ソルバー版のOptSeqでは利用できません．

```{.python.marimo}
#| export
def optseq_project_excel(res_list):

    R = []
    for i,r in enumerate(res_list):
        R.append(r["resource_name"])

    wb = Workbook()
    ws = wb.active
    wb.remove(ws)
    ws = wb.create_sheet(title="作業")
    ws.append(["作業ID", "作業名", "納期(年-月-日　時:分)", "後詰め(0,1)","遅れ重み（正数）", "後続作業ID（カンマ区切り）",
               "作業時間（分；整数）", "分割(0,1)", "並列(0,1)" ]+R 
              )
    #時間フォーマット
    for i in range(1000):
        ws.cell(2+i, 3).number_format = 'yyyy/m/d\\ h:mm;@'

    # #日付・時刻バリデーション
    dv = DataValidation(type="time") 
    ws.add_data_validation(dv)
    dv.add('C2:C1048576')   


    dv_list = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
    ws.add_data_validation(dv_list)
    dv_list.add('D2:D1048576')


    dv = DataValidation(type="whole",
                        operator="greaterThanOrEqual",
                        formula1=0)
    dv.add('E2:E1048576') 
    ws.add_data_validation(dv)


    dv = DataValidation(type="whole",
                        operator="greaterThanOrEqual",
                        formula1=0)
    dv.add('G2:G1048576') 
    ws.add_data_validation(dv)


    dv_list = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
    ws.add_data_validation(dv_list)
    dv_list.add('H2:I1048576')

    dv = DataValidation(type="whole",
                        operator="greaterThanOrEqual",
                        formula1=0)
    dv.add('J2:K1048576') 
    ws.add_data_validation(dv)   


    #コメント
    ws.cell(1,1).comment = Comment("作業ごとに異なる番号を入力", "logopt")
    ws.cell(1,2).comment = Comment("作業名", "logopt")
    ws.cell(1,3).comment = Comment("納期（時刻日付型）；ない場合は空白", "logopt")
    ws.cell(1,4).comment = Comment("作業をなるべく遅く実行するとき1", "logopt")
    ws.cell(1,5).comment = Comment("納期遅れペナルティ（1分あたりの費用）", "logopt")
    ws.cell(1,6).comment = Comment("後続作業のIDを半角カンマ区切りで入力", "logopt")
    #ws.cell(1,6).comment = Comment("作業を行う方法（モード）；作業関連は空白で必要数を行に追加", "logopt")
    ws.cell(1,7).comment = Comment("作業時間（分単位で非負の整数値）", "logopt")
    ws.cell(1,8).comment = Comment("作業の分割が可能なとき1", "logopt")
    ws.cell(1,9).comment = Comment("作業の並列実行が可能なとき1", "logopt")
    for i in range(len(res_list)):
        ws.cell(1,10+i).comment = Comment("対応する資源の量を非負整数で入力", "logopt")

    return wb
```

```{.python.marimo}
_res_list =[ {"resource_name": "機械", "seq_dependent": True}, {"resource_name": "作業員", "seq_dependent": False} ] 
_wb = optseq_project_excel(_res_list)
#wb.save("optseq-project.xlsx")
```

## 生産スケジューリング optseq_production_excel

注）この関数はSCMOPTに含まれるスケジューリング最適化システム用関数です．ソルバー版のOptSeqでは利用できません．

生産スケジューリングの場合のExcelテンプレートを生成する関数

返値：

- Excel Workbook

```{.python.marimo}
#| export
def optseq_production_excel():

    wb = Workbook()
    ws = wb.active
    wb.remove(ws)
    ws = wb.create_sheet(title="作業モード")
    ws.append(["作業名(ID)", "最早開始(年-月-日　時:分)","納期(年-月-日　時:分)", "後詰め(0,1)：1のときには必ず納期を設定","遅れ重み（正数）", "モード名（複数行も可）",
               "作業時間（分；非負整数）", "分割可(0,1)", "並列可(0,1)" ]
              )
    #時間フォーマット
    for i in range(10000):
        ws.cell(2+i, 2).number_format = 'yyyy/m/d\\ h:mm;@'
        ws.cell(2+i, 3).number_format = 'yyyy/m/d\\ h:mm;@'

    # #日付・時刻バリデーション
    dv = DataValidation(type="time") 
    ws.add_data_validation(dv)
    dv.add('B2:C1048576')   

    #コメント
    ws.cell(1,1).comment = Comment("作業名；固有の名称を入力", "logopt")
    ws.cell(1,2).comment = Comment("最早開始時刻（時刻日付型）；ない場合は空白", "logopt")
    ws.cell(1,3).comment = Comment("納期（時刻日付型）；ない場合は空白", "logopt")
    ws.cell(1,4).comment = Comment("作業をなるべく遅く実行するとき1（後詰めの作業の後続作業も後詰めになる）", "logopt")
    ws.cell(1,5).comment = Comment("納期遅れペナルティ（遅れ1分あたりの費用）", "logopt")
    ws.cell(1,6).comment = Comment("作業を行う方法（モード）；複数の場合は作業関連の列は空白で行を追加", "logopt")
    ws.cell(1,7).comment = Comment("作業時間（分単位で非負の整数値）", "logopt")
    ws.cell(1,8).comment = Comment("作業の分割が可能なとき1", "logopt")
    ws.cell(1,9).comment = Comment("作業の並列実行が可能なとき1", "logopt")


    #データバリデーション(TODO)


    dv_list = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
    ws.add_data_validation(dv_list)
    dv_list.add('D2:D1048576')


    dv = DataValidation(type="whole",
                        operator="greaterThanOrEqual",
                        formula1=0)
    dv.add('E2:E1048576') 
    ws.add_data_validation(dv)


    dv = DataValidation(type="whole",
                        operator="greaterThanOrEqual",
                        formula1=0)
    dv.add('G2:G1048576') 
    ws.add_data_validation(dv)


    dv_list = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
    ws.add_data_validation(dv_list)
    dv_list.add('H2:I1048576')

    #資源マスタ
    ws = wb.create_sheet(title="資源")
    ws.append(["資源名(ID)", "資源使用可能量上限（基準値）"])
    ws.cell(1,1).comment = Comment("資源名；固有の名称を入力", "logopt")
    ws.cell(1,2).comment = Comment("資源量上限の基準値を入力；日別・時間帯別の上限は後で変更可能", "logopt")

    #データバリデーション(TODO)
    dv = DataValidation(type="whole",
                        operator="greaterThanOrEqual",
                        formula1=1)
    dv.add('B2:B1048576') 
    ws.add_data_validation(dv)

    #時間制約
    ws = wb.create_sheet(title="時間制約")
    ws.append(["先行作業名", "後続作業名", "時間制約タイプ", "遅れ（分）" ])
    #コメント
    ws.cell(1,1).comment = Comment("先行作業名を作業モードシートの作業名から選択", "logopt")
    ws.cell(1,2).comment = Comment("後続作業名を作業モードシートの作業名から選択", "logopt")
    ws.cell(1,3).comment = Comment("完了・開始(CS)，開始・開始(SS)，完了・完了(CC)，開始・完了(SC)から選択", "logopt")
    ws.cell(1,4).comment = Comment("先行作業と後続作業の時間差を入力（負も可）", "logopt")

    dv_list = DataValidation(type="list", formula1='"CS,CC,SS,SC"', allow_blank=True)
    ws.add_data_validation(dv_list)
    dv_list.add('C2:C1048576')

    dv = DataValidation(type="whole")
    dv.add('D2:D1048576') 
    ws.add_data_validation(dv)


    #モード・資源制約
    ws = wb.create_sheet(title="資源使用量")
    ws.append(["モード名(ID)", "資源名(ID)", "開始時間", "終了時間", "使用量"])
    #コメント
    ws.cell(1,1).comment = Comment("モードの名称(ID)を入力", "logopt")
    ws.cell(1,2).comment = Comment("資源の名称(ID)を入力；複数の場合はモード名は空白で必要数を行に追加", "logopt")
    ws.cell(1,3).comment = Comment("作業開始時から資源使用開始時までの経過時間（分）;空白ならすべて", "logopt")
    ws.cell(1,4).comment = Comment("作業開始時から資源使用終了時までの経過時間（分）;空白ならすべて", "logopt")
    ws.cell(1,5).comment = Comment("資源の使用量を正数値で入力", "logopt")


    #データバリデーション(TODO)
    dv = DataValidation(type="whole",
                        operator="greaterThanOrEqual",
                        formula1=1)
    dv.add('C2:E1048576') 
    ws.add_data_validation(dv)   

    return wb
```

### optseq_production_excel関数の使用例

```{.python.marimo}
_wb = optseq_production_excel()
_wb.save('optseq-production.xlsx')
```

## 生産スケジューリング用の資源Excelファイル生成関数 optseq_resource_excel

注）この関数はSCMOPTに含まれるスケジューリング最適化システム用関数です．ソルバー版のOptSeqでは利用できません．

日タイプ別の資源上限入力用のExcelファイルの生成

資源は，カレンダー（休日），平日稼働時間，土曜稼働時間，休日稼働時間などを入れてデータベースに保管

それをもとに，作業シートを生成してダウンロード，その後記入して，アップロード

day_dfで計画期間内の日と日タイプの紐付けを行い，資源量上限を設定し，最適化

引数

- wb: OptSeq基本Workbook
- start: 開始日
- finish: 終了日
- period: 期を構成する単位期間の数；既定値は $1$
- period_unit: 期の単位 （時，日，週，月から選択）； 既定値は日； periodとあわせて期間の生成に用いる． たとえば，既定値だと１日が1期となる．


返値：

- wb: 日別の資源使用可能量条件設定用のシートを加えたWorkbook

```{.python.marimo}
#| export
def optseq_resource_excel(wb, start, finish, period=1, period_unit="時"):
    try:
        data = wb["資源"].values
        cols = next(data)[:]
        data = list(data)
        resource_df = pd.DataFrame(data, columns=cols).dropna(how="all") 
    except:
        raise KeyError("資源シートなし")

    res_list = list(resource_df.iloc[:,0])
    ub = list(resource_df.iloc[:,1])

    trans = {"秒":"S","分":"min","時":"h"}
    freq = f"{period}{trans[period_unit]}"

    #終了が同時か早い場合には１日加える
    start = pd.to_datetime(start)
    finish = pd.to_datetime(finish)
    if start>=finish:
        finish = finish + dt.timedelta(days=1)

    period_df = pd.DataFrame(pd.date_range(start, finish, freq=freq),columns=["description"])
    period_df["description"] = period_df.description.dt.strftime("%H:%M:%S")
    period_df["id"] = [t for t in range(len(period_df))]
    period_df = period_df.reindex(columns = ["id", "description"])

    T = len(period_df)-1
    day_type_list =['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Holiday']+[ f"Special{i}" for i in range(1,4)]
    wb = Workbook()
    ws = wb.active
    wb.remove(ws)
    for d in day_type_list:
        ws = wb.create_sheet(d)
        ws.append( ["資源名"]+list(period_df.description[:-1]) )
        for i, res in enumerate(res_list):
            ws.append([res] +[ub[i] for t in range(T)])
        #コメント
        ws.cell(1,1).comment = Comment("資源の名称(ID)を　入力", "logopt")
        ws.cell(1,2).comment = Comment("時刻ごとの資源使用可能量上限", "logopt")
    return wb
```

### optseq_resource_excel関数の使用例

```{.python.marimo}
_basic_wb = load_workbook('optseq-production.xlsx')
_period_unit = '時'
_period = 1
_start = '0:00'
_finish = '0:00'
_wb = optseq_resource_excel(_basic_wb, _start, _finish, _period, _period_unit)
_wb.save('optseq-resource.xlsx')
```

## ExcelのWorkbookをデータフレームに変換する関数 prepare_df_for_optseq

注）この関数はSCMOPTに含まれるスケジューリング最適化システム用関数です．ソルバー版のOptSeqでは利用できません．

引数：

- wb: OptSeqの基本Workbookオブジェクト
- resource_wb: 日タイプ別の資源使用可能量上限を入れたWorkbook
- day_wb: 日情報を入れたWorkbook

返値：

- act_df : 作業データフレーム
- time_df : 時間制約データフレーム
- usage_df: 資源使用量データフレーム
- resource_df_dic : 日タイプごとの資源使用可能量上限のデータフレームを入れた辞書
- day_df: 日データフレーム

```{.python.marimo}
#| export
def time_delta(finish, start):
    """
    日付時刻もしくは時刻型の差を計算して，秒を返す関数
    """
    try: #datetime型
        return int((finish-start).total_seconds())
    except TypeError: #time型
        td = (dt.datetime.combine(dt.date(2000,1,1), finish) - dt.datetime.combine(dt.date(2000,1,1), start) )
        return td.days*60*60*24 + td.seconds

def prepare_df_for_optseq(wb, resource_wb, day_wb): 
    #基本wbのシートの読み込みとデータフレームの準備
    data = wb["作業モード"].values
    cols = next(data)[:]
    data = list(data)
    act_df = pd.DataFrame(data, columns=cols).dropna(how="all") 
    data = wb["資源"].values
    cols = next(data)[:]
    data = list(data)
    resource_df = pd.DataFrame(data, columns=cols).dropna(how="all") 
    data = wb["時間制約"].values
    cols = next(data)[:]
    data = list(data)
    time_df = pd.DataFrame(data, columns=cols).dropna(how="all") 
    data = wb["資源使用量"].values
    cols = next(data)[:]
    data = list(data)
    usage_df = pd.DataFrame(data, columns=cols).dropna(how="all")

    #日タイプ別の資源データの準備
    day_type_list =['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Holiday']+[ f"Special{i}" for i in range(1,4)]
    res_df_dic ={}
    for d in day_type_list:
        data = resource_wb[d].values
        cols = next(data)[:]
        data = list(data)

        res_df_dic[d] = pd.DataFrame(data, columns=cols).dropna(how="all") 

    #各日の資源データの準備
    data = day_wb.active.values
    cols = next(data)[:]
    data = list(data)
    day_df = pd.DataFrame(data, columns=cols).dropna(how="all") 

    return  act_df, resource_df, time_df, usage_df, res_df_dic, day_df
```

### prepare_df_for_optseq関数の使用例

```{.python.marimo}
# _wb = load_workbook('optseq-master-ex1.xlsx')
# _resource_wb = load_workbook('optseq-resource-ex1.xlsx')
# _day_wb = load_workbook('optseq-day-ex1.xlsx')
# _act_df, _resource_df, _time_df, _usage_df, _res_df_dic, _day_df = prepare_df_for_optseq(_wb, _resource_wb, _day_wb)
```

## 期ごとの資源使用可能量上限を準備する関数 prepare_capacity

注）この関数はSCMOPTに含まれるスケジューリング最適化システム用関数です．ソルバー版のOptSeqでは利用できません．

引数：

- res_df_dic: 日タイプごとの資源使用可能量上限のデータフレームを入れた辞書
- day_df: 日データフレーム
- start: 基準時刻

返値：

- capacity: 資源使用可能量上限を入れた辞書

```{.python.marimo}
#| export
def prepare_capacity(res_df_dic, day_df, start):
    capacity = defaultdict(dict) #資源使用可能量上限
    for row in day_df.itertuples():
        day = str(row[1])
        day_type = str(row[2])
        #print(day, day_type)
        time_list = [ ]
        for i in res_df_dic[day_type].columns[1:]:
            t = pd.to_datetime(f"{day} {i}")
            time_list.append( time_delta(t,start)//60 )
        #print(time_list)
        T = len(time_list)
        for row2 in res_df_dic[day_type].itertuples():
            res_name = str(row2[1]) #TODO check resource! 
            st_time = time_list[0] #開始時刻
            usage = list( row2[2:])
            #print(usage)
            current = usage[0]
            for t in range(1,T):
                if usage[t] != current: #資源量上限が変化
                    capacity[res_name].update({(st_time,time_list[t]):current})
                    st_time = time_list[t]
                    current = usage[t]
            capacity[res_name].update({(st_time,time_list[T-1]+60):current})  #最後の時刻+60分が終了時刻
    return capacity
```

### prepare_capacity関数の使用例

```{.python.marimo}
# start_3 = dt.datetime(2021, 1, 1, 0, 0)
# capacity = prepare_capacity(res_df_dic, day_df, start_3)
# print(capacity)
```

## 生産スケジューリング用のExcelファイルを読み込んでモデルを生成する関数 make_model_for_optseq_production

注）この関数はSCMOPTに含まれるスケジューリング最適化システム用関数です．ソルバー版のOptSeqでは利用できません．

引数：

- act_df : 作業データフレーム
- resource_df : 資源データフレーム
- time_df : 時間制約データフレーム
- usage_df: 資源使用量データフレーム
- capacity: 資源使用可能量上限を入れた辞書
- start: 基準時刻
- fix_start: 作業の開始時刻の固定情報
- fix_finish: 作業の終了時刻の固定情報

返値：

- model: OptSeqモデルオブジェクト

```{.python.marimo}
def make_model_for_optseq_production(act_df, resource_df, time_df, usage_df, capacity, start, fix_start=None, fix_finish=None):
    model = Model()
    act, mode, res = ({}, {}, {})
    mode_id = 0
    for row in act_df.itertuples():
        if pd.isnull(row[1]) == False:
            act_name = str(row[1])
            if pd.isnull(row[2]) == False:
                release = time_delta(row[2], start) // 60
                if release < 0:
                    raise ValueError(f'作業名 {act_name} の開始時刻が早すぎます．基準時刻を再設定してください．')
            else:
                release = 0
            if pd.isnull(row[3]) == False:
                duedate = time_delta(row[3], start) // 60
                if duedate < 0:
                    raise ValueError(f'作業名 {act_name} の開始時刻が早すぎます．基準時刻を再設定してください．')
            else:
                duedate = 'inf'
            if pd.isnull(row[4]) == False:
                backward = int(row[4])
            else:
                backward = False
            if pd.isnull(row[5]) == False:
                weight = int(row[5])
            else:
                weight = 1
            act[act_name] = model.addActivity(name=act_name, duedate=duedate, backward=backward, weight=weight)
            if release > 0:
                model.addTemporal('source', act[act_name], delay=release)
        if pd.isnull(row[6]) == False:
            mode_name = str(row[6])
        else:
            mode_name = f'Dummy{mode_id}'
            mode_id = mode_id + 1
        if pd.isnull(row[7]) == False:
            duration = int(row[7])
        else:
            duration = 0
        mode[mode_name] = Mode(name=mode_name, duration=duration)
        if pd.isnull(row[8]) == False:
            if int(row[8]) > 0:
                mode[mode_name].addBreak(0, 'inf')
        if pd.isnull(row[9]) == False:
            if int(row[9]) > 0:
                mode[mode_name].addParallel(1, 'inf')
        act[act_name].addModes(mode[mode_name])
    if fix_start is not None:
        for act_name in fix_start:
            if fix_start[act_name] < 0:
                raise ValueError(f'作業名 {act_name} の開始時刻が早すぎます．基準時刻を再設定してください．')
            if act_name not in act:
                raise KeyError(f'固定すべき作業名 {act_name} がデータに含まれていません．')
            model.addTemporal('source', act[act_name], tempType='SS', delay=fix_start[act_name])
            model.addTemporal(act[act_name], 'source', tempType='SS', delay=-fix_start[act_name])
    if fix_finish is not None:
        for act_name in fix_finish:
            if fix_start[act_name] < 0:
                raise ValueError(f'作業名 {act_name} の終了時刻が早すぎます．基準時刻を再設定してください．')
            model.addTemporal('source', act[act_name], tempType='SC', delay=fix_finish[act_name])
            model.addTemporal(act[act_name], 'source', tempType='CS', delay=-fix_finish[act_name])
    for row in resource_df.itertuples():
        res_name = str(row[1])
        res[res_name] = model.addResource(name=res_name, capacity=capacity[res_name])
    for row in time_df.itertuples():
        pred, succ = (str(row[1]), str(row[2]))
        if pred not in act or succ not in act:
            raise KeyError(f'作業名 {pred} or {succ} が作業モードシートにありません')
        if pd.isnull(row[3]) == False:
            _type = str(row[3])
            if _type not in {'CS', 'CC', 'SS', 'SC'}:
                raise ValueError('制約タイプが未定です．')
        else:
            _type = 'CS'
        if pd.isnull(row[4]):
            delay = 0
        else:
            delay = int(row[4])
        model.addTemporal(pred=act[pred], succ=act[succ], tempType=_type, delay=delay)
    for row in usage_df.itertuples():
        mode_name = str(row[1])
        if mode_name not in mode:
            raise KeyError(f'モード名 {mode_name} が作業モードシートにありません')
        res_name = str(row[2])
        if res_name not in res:
            raise KeyError(f'資源名 {res_name} が資源シートにありません')
        if pd.isnull(row[3]) == False:
            st_time = int(row[3])
        else:
            st_time = 0
        if pd.isnull(row[4]) == False:
            fi = int(row[4])
        else:
            fi = 'inf'
        if pd.isnull(row[5]) == False:
            usage = int(row[5])
        else:
            raise ValueError('資源使用量が未定義です')
        mode[mode_name].addResource(res[res_name], {(st_time, fi): usage})
    return model
```

### make_model_for_optseq_production関数の使用例

```{.python.marimo}
# wb_4 = load_workbook('optseq-master-ex1.xlsx')
# resource_wb_1 = load_workbook('optseq-resource-ex1.xlsx')
# day_wb_1 = load_workbook('optseq-day-ex1.xlsx')
# act_df_1, resource_df_1, time_df_1, usage_df_1, res_df_dic_1, day_df_1 = prepare_df_for_optseq(wb_4, resource_wb_1, day_wb_1)
# start_4 = dt.datetime(2021, 1, 1, 0, 0)
# capacity_1 = prepare_capacity(res_df_dic_1, day_df_1, start_4)
# model_7 = make_model_for_optseq_production(act_df_1, resource_df_1, time_df_1, usage_df_1, capacity_1, start_4)
```

## 結果Workbookから固定情報の抽出関数　extract_fix_optseq

注）この関数はSCMOPTに含まれるスケジューリング最適化システム用関数です．ソルバー版のOptSeqでは利用できません．

引数：

- wb: ガントチャートのWorkbookオブジェクト（作業の開始時刻と終了時刻の固定情報が入っている）
- start: 基準時刻


返値：

- fix_start: 作業の開始時刻の固定情報
- fix_finish: 作業の終了時刻の固定情報

```{.python.marimo}
#| export
def extract_fix_optseq(wb, start):
    ws = wb.active
    fix_start, fix_finish = {}, {}
    for row in ws.iter_rows(min_row=3, min_col=1, max_col=3):
        if row[0] is None:
            break
        cell = row[1]
        if cell.fill.fgColor.rgb != "00000000": #白以外の色の行を抽出
            val = time_delta(cell.value, start)//60 #分
            fix_start[str(row[0].value)] = val
        cell = row[2]
        if cell.fill.fgColor.rgb != "00000000": #白以外の色の行を抽出
            val = time_delta(cell.value, start)//60
            fix_finish[str(row[0].value)] = val
    return fix_start, fix_finish
```

### extract_fix_optseq関数の使用例

```{.python.marimo}
# start_5 = dt.datetime(2021, 1, 1, 0, 0)
# wb_5 = load_workbook('optseq-result-ex1.xlsx')
# fix_start, fix_finish = extract_fix_optseq(wb_5, start_5)
# (fix_start, fix_finish)
```

## 生産スケジューリング用のガントチャート生成関数 make_gantt_for_production

注）この関数はSCMOPTに含まれるスケジューリング最適化システム用関数です．ソルバー版のOptSeqでは利用できません．

引数：

- model: OptSeqモデルインスタンス（最適化後）
- capacity: 資源使用可能量上限を入れた辞書
- start: 開始日

返値：

- wb: ガントチャートのWorkbook

```{.python.marimo}
def make_gantt_for_production(model, capacity, start='2020/1/1 00:00:00'):
    """
    Excelのガントチャートを生成する関数
    """
    MIN_COL = 8

    def prepare_res_idx():
        count = 0
        name_list = []
        for res in model.res:
            if len(res.terms) == 0:
                count = count + 1
                name_list.append(res.name)
        n_res = len(name_list)
        if n_res > 0:
            name_dic = {}
            for i, name in enumerate(name_list):
                name_dic[name] = i + 1
            res_idx_list = []
            for idx, i in enumerate(model.activities):
                a = model.activities[i]
                try:
                    res_name = list(a.selected.requirement.keys())[0][0]
                    res_idx = name_dic[res_name]
                except IndexError:
                    res_idx = n_res + 1
                res_idx_list.append(res_idx)
        return (res_idx_list, name_list)

    def time_convert_long(periods, period='minutes'):
        if period == 'days':
            time_ = start + dt.timedelta(days=float(periods))
        elif period == 'hours':
            time_ = start + dt.timedelta(hours=float(periods))
        elif period == 'minutes':
            time_ = start + dt.timedelta(minutes=float(periods))
        elif period == 'seconds':
            time_ = start + dt.timedelta(seconds=float(periods))
        else:
            raise TypeError("pariod must be 'days' or 'seconds' or minutes' or 'days'")
        return time_.strftime('%Y/%m/%d %H:%M')
    start = pd.to_datetime(start)
    n_job = len(model.activities)
    wb = Workbook()
    ws = wb.active
    max_time = 0
    for i in model.activities:
        a = model.activities[i]
        max_time = max(max_time, int(a.completion))
        if a.duedate != 'inf':
            max_time = max(max_time, int(a.duedate))
    max_time = max_time // 60
    ws.append(['基準日', start, ' ', '  ', ' ', '    ', '    '])
    for j in range(max_time):
        cell = ws.cell(1, j + 8)
        col = cell.column_letter
        cell.value = f'=LEFT(TEXT({col}2,"aaaa"),1)'
    ws.append(['名称', '開始', '終了', 'モード', '作業時間', '納期', '遅れ重み'] + list(pd.date_range(start=start, periods=max_time, freq='h')))
    for idx, i in enumerate(model.activities):
        L = []
        a = model.activities[i]
        st = time_convert_long(a.start)
        fi = time_convert_long(a.completion)
        if a.duedate != 'inf':
            due = time_convert_long(a.duedate)
        else:
            due = None
        L = [a.name, pd.to_datetime(st), pd.to_datetime(fi), a.selected.name, a.selected.duration, due, a.weight]
        ws.append(L)
    for idx, i in enumerate(model.activities):
        a = model.activities[i]
        if a.duedate != 'inf':
            side = Side(style='thick', color='00FF0000')
            ws.cell(idx + 3, 7 + int(a.duedate // 60)).border = Border(right=side)
    _format = 'yyyy/m/d\\ h:mm;@'
    ws.cell(1, 2).number_format = _format
    for j in range(max_time):
        ws.cell(2, MIN_COL + j).number_format = _format
    for i in range(n_job):
        ws.cell(i + 3, 2).number_format = _format
        ws.cell(i + 3, 3).number_format = _format
    res_idx_list, name_list = prepare_res_idx()
    ws.append(['', '', '', '', ' ', '資源ID', '資源名/期'] + list(range(max_time)))
    for j, res in enumerate(name_list):
        ws.append(['', '', '', '', '', j + 1, res + '（使用量）'])
    for j, res in enumerate(name_list):
        for t in range(max_time):
            cell = ws.cell(n_job + 4 + j, MIN_COL + t)
            col = cell.column_letter
            cell.value = f'=COUNTIF({col}3:{col}{n_job + 2},{j + 1})'
    ws.append(['', '', '', '', '', '資源ID', '資源名'])
    for j, m in enumerate(capacity):
        L = [0 for t in range(max_time)]
        for st, fi in capacity[m]:
            s = st // 60
            f = min(fi // 60, max_time)
            for i in range(s, f):
                L[i] = L[i] + capacity[m][st, fi]
        ws.append(['', '', '', '', '', j + 1, str(m) + '(上限）'] + L)
    for i in range(n_job):
        for j in range(max_time):
            cell = ws.cell(i + 3, MIN_COL + j)
            col = cell.column_letter
            cell.value = f'=IF(AND({col}2<C{i + 3},{col}2>=B{i + 3}),{res_idx_list[i]},0)'
    redFill = PatternFill(start_color='0099CC', end_color='0099CC', fill_type='solid')
    cell = ws.cell(3, MIN_COL + max_time)
    col = cell.column_letter
    ws.conditional_formatting.add(f'H3:{col}{2 + n_job}', FormulaRule(formula=['H3>=1'], fill=redFill))
    for j in range(max_time):
        cell = ws.cell(2, MIN_COL + j)
        cell.alignment = Alignment(horizontal='center', vertical='center', textRotation=90, wrap_text=False)
    ws.row_dimensions[2].height = 100
    ws.column_dimensions['B'].width = 15
    ws.column_dimensions['C'].width = 15
    ws.column_dimensions['F'].width = 15
    ws.column_dimensions['G'].width = 15
    for j in range(max_time):
        cell = ws.cell(1, MIN_COL + j)
        col = cell.column_letter
        ws.column_dimensions[col].width = 3
    for j, res in enumerate(name_list):
        c1 = ScatterChart()
        c1.title = f'{j + 1}. 資源 {res}'
        c1.style = 13
        c1.y_axis.title = '資源量'
        c1.x_axis.title = '期'
        xvalues = Reference(ws, min_col=MIN_COL - 1, min_row=n_job + 3, max_col=7 + max_time)
        yvalues1 = Reference(ws, min_col=MIN_COL - 1, min_row=n_job + 4 + j, max_col=7 + max_time)
        series1 = Series(yvalues1, xvalues, title_from_data=True)
        series1.marker.symbol = 'triangle'
        series1.marker.graphicalProperties.solidFill = 'FF0000'
        series1.marker.graphicalProperties.line.solidFill = 'FF0000'
        series1.graphicalProperties.line.noFill = True
        c1.series.append(series1)
        yvalues2 = Reference(ws, min_col=MIN_COL - 1, min_row=n_job + 5 + len(name_list) + j, max_col=7 + max_time)
        series2 = Series(yvalues2, xvalues, title_from_data=True)
        series2.graphicalProperties.line.solidFill = '00AAAA'
        series2.graphicalProperties.line.dashStyle = 'sysDot'
        series2.graphicalProperties.line.width = 100050
        c1.series.append(series2)
        ws.add_chart(c1, f'A{7 + n_job + len(name_list) * 2 + j * 20}')
    return wb
```

### make_gantt_for_production関数の使用例

```{.python.marimo}
# model.Params.TimeLimit=3
# model.optimize()
# wb = make_gantt_for_production(model, capacity, start="2021/1/1 00:00:00")
# wb.save("optseq-gantt-ex1.xlsx")
```

```{.python.marimo}
import marimo as mo
```