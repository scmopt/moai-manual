---
title: Scop
marimo-version: 0.10.12
width: full
---

# 制約最適化システム SCOP

>  Coonstraint Programming Solver SCOP
<!---->
<p> <a class="reference external" href="https://colab.research.google.com/github/scmopt/moai-manual/blob/main/scop-trial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> <a class="reference external" href="https://studiolab.sagemaker.aws/import/github/scmopt/moai-manual/blob/main/scop-trial.ipynb"><img alt="Open In SageMaker Studio Lab" src="https://studiolab.sagemaker.aws/studiolab.svg" /></a></p>
<!---->
SCOP（Solver for COnstraint Programing：スコープ）は，
大規模な制約最適化問題を高速に解くためのソルバーである．

ここで，制約最適化(constraint optimization)とは，
数理最適化を補完する最適化理論の体系であり，
組合せ最適化問題に特化した求解原理-メタヒューリスティクス(metaheuristics)-を用いるため，
数理最適化ソルバーでは求解が困難な大規模な問題に対しても，効率的に良好な解を探索することができる．

SCOPのトライアル・バージョンは， http://logopt.com/scop2/ からダウンロードするか，以下のように pipコマンドでインストールできる．

```python
pip install scop
```

```{.python.marimo disabled="true" hide_code="true"}
#| export

#Pydantic
from typing import List, Optional, Union, Tuple, Dict, Set, Any, DefaultDict, ClassVar
from pydantic import (BaseModel, Field, ValidationError, validator, 
                      confloat, conint, constr, Json, PositiveInt, NonNegativeInt)
from pydantic.tools import parse_obj_as
from datetime import datetime, date, time

import os
import sys
import re
import copy
import platform
import string
trans = str.maketrans(":-+*/'(){}^=<>$ |#?,\¥", "_"*22) #文字列変換用
import ast
import pickle
import datetime as dt
from collections import Counter
import pathlib

#以下非標準ファイル
#import pandas as pd
#import numpy as np
import plotly.graph_objs as go
import plotly
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
```

## 重み付き制約充足問題

ここでは，SCOPで対象とする重み付き制約充足問題について解説する．

一般に**制約充足問題**(constraint satisfaction problem)は，以下の3つの要素から構成される．

- 変数(variable): 分からないもの，最適化によって決めるもの．
制約充足問題では，変数は，与えられた集合（以下で述べる「領域」）から1つの要素を選択することによって決められる．

- 領域(domain): 変数ごとに決められた変数の取り得る値の集合

- 制約(constraint): 幾つかの変数が同時にとることのできる値に制限を付加するための条件．
SCOPでは線形制約（線形式の等式，不等式），2次制約（一般の2次式の等式，不等式），
相異制約（集合に含まれる変数がすべて異なることを表す制約）が定義できる．

制約充足問題は，制約をできるだけ満たすように，
変数に領域の中の1つの値を割り当てることを目的とした問題である．


SCOPでは，**重み付き制約充足問題**(weighted constraint satisfaction problem)
を対象とする．

ここで「制約の重み」とは，制約の重要度を表す数値であり，
SCOPでは正数値もしくは無限大を表す文字列 'inf'を入力する．
'inf'を入力した場合には，制約は**絶対制約**(hard constraint)とよばれ，
その逸脱量は優先して最小化される．
重みに正数値を入力した場合には，制約は**考慮制約**(soft constraint)とよばれ，
制約を逸脱した量に重みを乗じたものの和の合計を最小化する．

すべての変数に領域内の値を割り当てたものを**解**(solution)とよぶ．
SCOPでは，単に制約を満たす解を求めるだけでなく，
制約からの逸脱量の重み付き和（ペナルティ）を最小にする解を探索する．
<!---->
## 数理最適化や周辺技術との違い

ここでは，SCOPと数理最適化ソルバー，2次無制約2値最適化ソルバーとの違いについて簡単に触れる．

数理最適化は，混合整数最適化と非線形最適化の分類され，実数，（2値を含む）整数の変数を用い，**数式**によってモデルを記述する．
混合整数最適化ソルバーの多くは，線形最適化を基礎とした分枝切除法による厳密解法である．
そのため記述能力は高いが，大規模な整数最適化問題では大量の計算資源を要することがある．

2次無制約2値最適化は，制約がない2値変数から成る2次最適化のみを扱う．この問題に特化した近似解法を用いることが多い．
そのため記述能力は低いが，2次2値最適化問題を高速に（近似解を許容するという意味で）解くことができる．

SCOPは，組合せ最適化に特化した解法を用いているため，実数変数を含んだ最適化問題は苦手である（整数変数に変換すれば解くことができる）．
一方，2値変数や整数変数だけでなく，任意の集合（領域）から選択する変数をもつため記述能力は高く，
大規模問題でも（近似解を許容するという意味で）解くことができる．
以下に，組合せ最適化問題に対する記述能力と対応可能な問題の規模を評価尺度としたときの各手法の適用範囲を図にしておく．

<img src="https://github.com/scmopt/scmopt_data/blob/main/scop1.png?raw=true" width=600 height=200>

SCOPの記述能力の高さの例として，シフト最適化問題を考える．シフト最適化問題とは，スタッフ（作業員）にシフト（勤務の種類）を割り当てる問題である．
シフトというのは，1日の勤務の種類を表す用語である，
最近では短時間のバイトも増えており，何時から何時まで何の業務を行うといった細かい単位のシフトを設定する必要が出てきている．
ここでは，100人のスタッフに100種類のシフトを割り当てることを考える．

これを数理最適化や2次無制約2値最適化で定式化するには，スタッフにシフトを割り当てることを表す2値（$0$-$1$）変数を用いる必要があるので，10000の変数が必要になる．
一方，SCOPで定式化する際には，スタッフ $i$ に対して領域をシフトとした変数を準備するだけで良いので100の変数で表現できる．
<!---->
## SCOPの基本クラス

SCOPモジュール (scop.py) は，以下のクラスから構成されている．

- モデルクラス Model
- 変数クラス Variable
- 制約クラス Constraint (これは，以下のクラスのスーパークラスである．）

  - 線形制約クラス Linear
  - 2次制約クラス Quadratic
  - 相異制約クラス Alldiff

<!-- ![](https://github.com/scmopt/scmopt_data/blob/main/fig/scopclass.jpg?raw=true "scop-class") -->

<img src="https://github.com/scmopt/scmopt_data/blob/main/fig/scopclass.jpg?raw=true" width=600 height=200>
<!---->
## 注意

SCOPでは数（制約）名や値を文字列で区別するため重複した名前を付けることはできない．
なお，使用できる文字列は, 英文字 (a--z, A--Z),
数字 (0--9), 大括弧 ([ ]), アンダーバー (_), および @ に限定される．

それ以外の文字列はすべてアンダーバー (_)に置き換えられる．
<!---->
## パラメータクラス Parameters

Parametersクラスで設定可能なパラメータは，以下の通り．


-    TimeLimit は制限時間を表す．制限時間は正数値を設定する必要があり，その既定値は 600秒である．

-    OutputFlag は出力フラグを表し，最適化の過程を出力する際の詳細さを制御するためのパラメータである．
真(True もしくは正の値)に設定すると詳細な情報を出力し，
偽(False もしくは 0)に設定すると最小限の情報を出力する．
既定値は偽(0)である．

-    RandomSeed は乱数の種である．SCOPでは探索にランダム性を加味しているので，乱数の種を変えると，得られる解が変わる可能性がある．
乱数の種の既定値は 1である．

-   Target は制約の逸脱量が目標値以下になったら自動終了させるためのパラメータである．
既定値は 0 である．

-  Initial は，前回最適化の探索を行った際の最良解を初期値とした探索を行うとき True ，それ以外のとき  False を表すパラメータである．
既定値は False である．最良解の情報は，「変数:値」を1行としたテキストとしてファイル名 scop_best_data.txt に保管されている．
このファイルを書き換えることによって，異なる初期解から探索を行うことも可能である．

```{.python.marimo}
# | export
class Parameters(BaseModel):
    """
    SCOP parameter class to control the operation of SCOP.

    - TimeLimit: Limits the total time expended (in seconds). Positive integer. Default = 600.
    - OutputFlag: Controls the output log. Boolean. Default = False.
    - RandomSeed: Sets the random seed number. Integer. Default = 1.
    - Target: Sets the target penalty value;
            optimization will terminate if the solver determines that the optimum penalty value
            for the model is worse than the specified "Target." Non-negative integer. Default = 0.
    - Initial: True if you want to solve the problem starting with an initial solution obtained before, False otherwise. Default = False.
    """

    TimeLimit: int = 600
    OutputFlag: bool = False
    RandomSeed: int = 1
    Target: int = 0
    Initial: bool = False

    def __str__(self):
        return f" TimeLimit = {self.TimeLimit} \n OutputFlag = {self.OutputFlag} \n RandomSeed = {self.RandomSeed} \n Taeget = {self.Target} \n Initial = {self.Initial}"
```

### Parametersクラスの使用例

```python
params = Parameters()
params.TimeLimit = 3
print(params)
```

出力
```python
 TimeLimit = 3
 OutputFlag = False
 RandomSeed = 1
 Taeget = 0
 Initial = False
```
<!---->
## 変数クラス Variable

変数クラス Variable のインスタンスは，モデルインスタンスの addVariable もしくは addVariables メソッドを
用いて生成される．

```python
  変数インスタンス=model.addVariable(name, domain）
```

引数の  name は変数名を表す文字列であり，
  domain は領域を表すリストである．

```python
  変数インスタンスのリスト=model.addVariables(names, domain）
```

引数の  names は変数名を要素としたリストであり，  domain は領域を表すリストである．


変数クラスは，以下の属性をもつ．


-    name は変数の名称である．
-    domain は変数の領域(domain)を表すリストである．変数には領域に含まれる値(value)のうちの1つが割り当てられる．
-    value は最適化によって変数に割り当てられた値である．最適化が行われる前には  None が代入されている．


また，変数インスタンスは，変数の情報を文字列として返すことができる．

```{.python.marimo disabled="true" hide_code="true"}
class Variable(BaseModel):
    """
    SCOP variable class. Variables are associated with a particular model.
    You can create a variable object by adding a variable to a model (using Model.addVariable or Model.addVariables)
    instead of by using a Variable constructor.
    """
    ID: ClassVar[int] = 0
    name: str = ''
    domain: List[Union[str, int]] = None
    value: Optional[str] = None

    def __init__(self, name='', domain=None):
        super().__init__(name=name, domain=domain)
        if name is None or name == '':
            name = '__x{0}'.format(Variable.ID)
            Variable.ID = Variable.ID + 1
        if type(name) != str:
            raise ValueError('Variable name must be a string')
        if domain is None:
            domain = []
        self.name = str(name).translate(trans)
        self.domain = [str(d) for d in domain]
        self.value = None

    def __str__(self):
        return 'variable {0}:{1} = {2}'.format(str(self.name), str(self.domain), str(self.value))
```

### Variableクラスの使用例

```python
#標準的な使用法
_var = Variable(name="X[1]", domain=[1,2,3])
print(_var)

#変数名を省略した場合
_var1 = Variable(domain=[1, 2, 3])
_var2 = Variable(domain=[4, 5, 6])
print(_var1)
print(_var2)

#Pydantic検証
try:
    _var1 = Variable(name=1, domain=[1, 2, 3])
except ValueError as error:
    print(error)
```

出力
```python
variable X[1]:['1', '2', '3'] = None

variable __x4:['1', '2', '3'] = None
variable __x5:['4', '5', '6'] = None

1 validation error for Variable
name
  Input should be a valid string [type=string_type, input_value=1, input_type=int]
    For further information visit https://errors.pydantic.dev/2.10/v/string_type
```
<!---->
## 制約クラス Constraint

制約クラスは，以下で定義する線形制約クラス，2次制約クラス，相異制約クラスの基底クラスである．

```{.python.marimo disabled="true" hide_code="true"}
class Constraint(BaseModel):
    """
     Constraint base class
    """
    ID: ClassVar[int] = 0
    name: Optional[str] = ''
    weight: Optional[int] = 1

    def __init__(self, name='', weight=1):
        super().__init__(name=name, weight=weight)
        if name is None or name == '':
            name = '__CON[{0}]'.format(Constraint.ID)
            Constraint.ID = Constraint.ID + 1
        if type(name) != str:
            raise ValueError('Constraint name must be a string')
        self.name = str(name).translate(trans)

    def setWeight(self, weight):
        self.weight = weight
```

## モデルクラス Model

PythonからSCOPをよび出して使うときに，最初にすべきことはモデルクラス Model のインスタンスを生成することである．
たとえば，  'test' と名付けたモデルインスタンス  model を生成したいときには，以下のように記述する．

```python
from scop import *
model = Model('test')
```

インスタンスの引数はモデル名であり，省略すると無名のモデルが生成される．

Model クラスは，以下のメソッドをもつ．


-    addVariable(name，domain) はモデルに1つの変数を追加する．
引数の name は変数名を表す文字列であり，domain は領域を表すリストである．
変数名を省略すると，自動的に  __x[ 通し番号  ] という名前が付けられる．

領域を定義するためのリスト domain の要素は，文字列でも数値でもかまわない．
（ただし内部表現は文字列であるので，両者は区別されない．）

以下に例を示す．
```python
x = model.addVarriable('var')                     # domain  is set to []
x = model.addVariable(name='var',domain=[1,2,3])  # arguments by name
x = model.addVariable('var',['A','B','C'])        # arguments by position
```

1行目の例では，変数名を  'var' と設定した空の領域をもつ変数を追加している．
2行目の例では，名前付き引数で変数名と領域を設定している．領域は  1,2,3 の数値である．
3行目の例では，領域を文字列として変数を追加している．


-    addVariables(names,domain) はモデルに，同一の領域をもつ複数の変数を同時に追加する．
引数の  names は変数名を要素としたリストであり，domain は領域を表すリストである．
領域を定義するためのリストの要素は，文字列でも数値でもかまわない．

-    addConstriant(con) は制約インスタンス  con をモデルに追加する．
制約インスタンスは，制約クラスを用いて生成されたインスタンスである．制約インスタンスの生成法については，
以下で解説する．

-    optimize はモデルの求解（最適化）を行うメソッドである．
最適化のためのパラメータは，パラメータ属性  Params で設定する．
返値は，最適解の情報を保管した辞書と，破った制約の情報を保管した辞書のタプルである．

たとえば，以下のプログラムでは最適解を辞書  sol に，破った制約を辞書  violated に保管する．

```python
sol,violated= model.optimize()
```

最適解や破った制約は，変数や制約の名前をキーとし，解の値や制約逸脱量を値とした辞書であるので，
最適解の値と逸脱量を出力するには，以下のように記述すれば良い．

```python
for x in sol:
    print(x, sol[x])
for v in violated:
    print(v,violated[v])
```

　引数：

　- cloud: 複数人が同時実行する可能性があるときTrue（既定値はFalse）; Trueのとき，ソルバー呼び出し時に生成されるファイルにタイムスタンプを追加し，計算終了後にファイルを消去する．


モデルインスタンスは，以下の属性をもつ．

-    name はモデルの名前である．コンストラクタの引数として与えられる．省略可で既定値は ' ' である．
-    variables は変数インスタンスのリストである．
-    constraints は制約インスタンスのリストである．
-    varDict は制約名をキーとし，変数インスタンスを値とした辞書である．
-    Params は求解（最適化）の際に用いるパラメータを表す属性を保管する．
-    Status は最適化の状態を表す整数である．状態の種類と意味を以下の表に示す．


最適化の状態を表す整数と意味

|  状態の定数   |  説明  |
| ---- | ---- |
|0                |  最適化成功  |
|1   |   求解中にユーザが  Ctrl-C を入力したことによって強制終了した．  |
|2   |   入力データファイルの読み込みに失敗した．    |
|3   |   初期解ファイルの読み込みに失敗した．  |
|4   |   ログファイルの書き込みに失敗した．  |
|5   |  入力データの書式にエラーがある．  |
|6   |  メモリの確保に失敗した． |
|7   |  実行ファイル  scop.exe のよび出しに失敗した．  |
|10   |  モデルの入力は完了しているが，まだ最適化されていない． |
|負の値  |  その他のエラー  |



また，モデルインスタンスは，モデルの情報を文字列として返すことができる．

```{.python.marimo disabled="true" hide_code="true"}
class Model(BaseModel):
    """
    SCOP model class.

    Attbibutes:
    - constraints: Set of constraint objects in the model.
    - variables: Set of variable objects in the model.
    - Params:  Object including all the parameters of the model.
    - varDict: Dictionary that maps variable names to the variable object.

    """
    name: Optional[str] = ''
    constraints: Optional[List[Constraint]] = []
    variables: Optional[List[Variable]] = []
    Params: Optional[Parameters] = Parameters()
    varDict: Optional[Dict[str, List]] = {}
    Status: Optional[int] = 10

    def __str__(self):
        """
            return the information of the problem
            constraints are expanded and are shown in a readable format
        """
        ret = ['Model:' + str(self.name)]
        ret.append('number of variables = {0} '.format(len(self.variables)))
        ret.append('number of constraints= {0} '.format(len(self.constraints)))
        for v in self.variables:
            ret.append(str(v))
        for c in self.constraints:
            ret.append('{0} :LHS ={1} '.format(str(c)[:-1], str(c.lhs)))
        return ' \n'.join(ret)

    def update(self):
        """
        prepare a string representing the current model in the scop input format
        """
        f = []
        for var in self.variables:
            domainList = ','.join([str(i) for i in var.domain])
            f.append('variable %s in { %s } \n' % (var.name, domainList))
        f.append('target = %s \n' % str(self.Params.Target))
        for con in self.constraints:
            f.append(str(con))
        return ' '.join(f)

    def addVariable(self, name='', domain=[]):
        """
        - addVariable ( name="", domain=[] )
          Add a variable to the model.

        Arguments:
        - name: Name for new variable. A string object.
        - domain: Domain (list of values) of new variable. Each value must be a string or numeric object.

        Return value:
        New variable object.

        Example usage:
        x = model.addVarriable("var")                     # domain  is set to []
        x = model.addVariable(name="var",domain=[1,2,3])  # arguments by name
        x = model.addVariable("var",["A","B","C"])        # arguments by position

        """
        var = Variable(name, domain)
        if var.name in self.varDict:
            raise ValueError("duplicate key '{0}' found in variable name".format(var.name))
        else:
            self.variables.append(var)
            self.varDict[var.name] = var
        return var

    def addVariables(self, names=[], domain=[]):
        """
        - addVariables(names=[], domain=[])
           Add variables and their (identical) domain.

        Arguments:
        - names: list of new variables. A list of string objects.
        - domain: Domain (list of values) of new variables. Each value must be a string or numeric object.

        Return value:
        List of new variable objects.

        Example usage:
        varlist=["var1","var2","var3"]
        x = model.addVariables(varlist)                      # domain  is set to []
        x = model.addVariables(names=varlist,domain=[1,2,3]  # arguments by name
        x = model.addVariables(varlist,["A","B","C"]         # arguments by position

        """
        if type(names) != type([]):
            raise TypeError('The first argument (names) must be a list.')
        varlist = []
        for var in names:
            varlist.append(self.addVariable(var, domain))
        return varlist

    def addConstraint(self, con):
        """
        addConstraint ( con )
        Add a constraint to the model.

        Argument:
        - con: A constraint object (Linear, Quadratic or AllDiff).

        Example usage:
        model.addConstraint(L)

        """
        if not isinstance(con, Constraint):
            raise TypeError('error: %r should be a subclass of Constraint' % con)
        try:
            if con.feasible(self.varDict):
                self.constraints.append(con)
        except NameError:
            raise NameError('Consrtaint %r has an error ' % con)

    def optimize(self, cloud=False):
        """
        optimize ()
        Optimize the model using scop.exe in the same directory.

        Example usage:
        model.optimize()
        """
        time = self.Params.TimeLimit
        seed = self.Params.RandomSeed
        LOG = self.Params.OutputFlag
        f = self.update()
        p = pathlib.Path('.')
        if cloud:
            input_file_name = f'scop_input{dt.datetime.now().timestamp()}.txt'
            f3 = open(input_file_name, 'w')
            script = p / 'scripts/scop'
        else:
            f3 = open('scop_input.txt', 'w')
            script = './scop'
        f3.write(f)
        f3.close()
        if LOG >= 100:
            print('scop input: \n')
            print(f)
            print('\n')
        if LOG:
            print('solving using parameters: \n ')
            print('  TimeLimit =%s second \n' % time)
            print('  RandomSeed= %s \n' % seed)
            print('  OutputFlag= %s \n' % LOG)
        import subprocess
        if platform.system() == 'Windows':
            cmd = 'scop -time ' + str(time) + ' -seed ' + str(seed)
        elif platform.system() == 'Darwin':
            if platform.mac_ver()[2] == 'arm64':
                cmd = f'{script}-m1 -time ' + str(time) + ' -seed ' + str(seed)
            else:
                cmd = f'{script} -time ' + str(time) + ' -seed ' + str(seed)
        elif platform.system() == 'Linux':
            cmd = f'{script}-linux -time ' + str(time) + ' -seed ' + str(seed)
        if self.Params.Initial:
            cmd = cmd + ' -initsolfile scop_best_data.txt'
        try:
            if platform.system() == 'Windows':
                pipe = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
            else:
                pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
            print('\n ================ Now solving the problem ================ \n')
            out, err = pipe.communicate(f.encode())
            if out == b'':
                raise OSError
        except OSError:
            print('error: could not execute command')
            print('please check that the solver is in the path')
            self.Status = 7
            return (None, None)
        if err != None:
            if int(sys.version_info[0]) >= 3:
                err = str(err, encoding='utf-8')
            f2 = open('scop_error.txt', 'w')
            f2.write(err)
            f2.close()
        if int(sys.version_info[0]) >= 3:
            out = str(out, encoding='utf-8')
        if cloud:
            os.remove(input_file_name)
        if LOG:
            print(out, '\n')
        if cloud:
            pass
        else:
            f = open('scop_out.txt', 'w')
            f.write(out)
            f.close()
        self.Status = pipe.returncode
        if self.Status != 0:
            print('Status=', self.Status)
            print('Output=', out)
            return (None, None)
        s0 = '[best solution]'
        s1 = 'penalty'
        s2 = '[Violated constraints]'
        i0 = out.find(s0) + len(s0)
        i1 = out.find(s1, i0)
        i2 = out.find(s2, i1) + len(s2)
        data = out[i0:i1].strip()
        if cloud:
            pass
        else:
            f3 = open('scop_best_data.txt', 'w')
            f3.write(data.lstrip())
            f3.close()
        sol = {}
        if data != '':
            for s in data.split('\n'):
                name, value = s.split(':')
                sol[name] = value.strip()
        data = out[i2:].strip()
        violated = {}
        if data != '':
            for s in data.split('\n'):
                try:
                    name, value = s.split(':')
                except:
                    print('Error String=', s)
                try:
                    temp = int(value)
                except:
                    violated[name] = value
                else:
                    violated[name] = int(value)
        for name in sol:
            if name in self.varDict:
                self.varDict[name].value = sol[name]
            else:
                raise NameError('Solution {0} is not in variable list'.format(name))
        for con in self.constraints:
            if isinstance(con, Linear):
                lhs = 0
                for coeff, var, domain in con.terms:
                    if var.value == domain:
                        lhs = lhs + coeff
                con.lhs = lhs
            if isinstance(con, Quadratic):
                lhs = 0
                for coeff, var1, domain1, var2, domain2 in con.terms:
                    if var1.value == domain1 and var2.value == domain2:
                        lhs = lhs + coeff
                con.lhs = lhs
            if isinstance(con, Alldiff):
                VarSet = set([])
                lhs = 0
                for v in con.varlist:
                    index = v.domain.index(v.value)
                    if index in VarSet:
                        lhs = lhs + 1
                    VarSet.add(index)
                con.lhs = lhs
        return (sol, violated)
```

### Modelクラスの使用例

```python
_model = Model(name='test')
_model.addVariable(name='x[1]', domain=[0, 1, 2])
print(_model)
_model.Params.TimeLimit = 1
_sol, _violated = _model.optimize()
print('solution=', _sol)
print('violated constraints=', _violated)
print('status="', _model.Status)
```

出力
```python
Model:test
number of variables = 1
number of constraints= 0
variable x[1]:['0', '1', '2'] = None

 ================ Now solving the problem ================

solution= {'x[1]': '0'}
violated constraints= {}
status = 0
```
<!---->
## 線形制約クラス Linear

最も基本的な制約は，線形制約である．
線形制約は

$$
  線形項1 + 線形項2 + \cdots  制約の方向 (\leq, \geq, =) 右辺定数
$$

の形で与えられる．線形項は，「変数」が「値」をとったとき $1$，
それ以外のとき $0$ を表す**値変数**(value variable) $x[変数,値]$ を用いて，

$$
 係数 \times x[変数,値]
$$

で与えられる．

ここで，係数ならびに右辺定数は整数とし，制約の方向は
以下($\leq$)，以上($\geq$)，等しい($=$)から1つを選ぶ必要がある．

線形制約クラス  Linear のインスタンスは，以下のように生成する．

```python
線形制約インスタンス=Linear(name, weight=1, rhs=0, direction='<=')
```


引数の意味は以下の通り．


-     name は制約の名前を表す．これは制約を区別するための名称であり，固有の名前を文字列で入力する必要がある．
（名前が重複した場合には，前に定義した制約が無視される．）
名前を省略した場合には，自動的に  __CON[ 通し番号  ] という名前が付けられる．
これは，以下の2次制約や相異制約でも同じである．

-    weight は制約の重みを表す．重みは，制約の重要性を表す正数もしくは文字列 'inf' である．
ここで 'inf' は無限大を表し，絶対制約を定義するときに用いられる．
重みは省略することができ，その場合の既定値は 1である．

-    rhs は制約の右辺定数(right hand side)を表す．
右辺定数は，制約の右辺を表す定数（整数値）である．
右辺定数は省略することができ，その場合の既定値は 0 である．

-    direction は制約の向きを表す．
制約の向きは，
  '<=' ,   '>=' ,   '=' のいずれかの文字列とする．既定値は  '<=' である．

上の引数はすべて Linear クラスの属性となる．最適化を行った後では，制約の左辺の評価値が属性として参照可能になる．


-    lhs は制約の左辺の値を表す．これは最適化によって得られた変数の値を左辺に代入したときの評価値である．
最適化を行う前には 0 が代入されている



  Linear クラスは，以下のメソッドをもつ．


-    addTerms(coeffs, vars，values) は，線形制約の左辺に1つもしくは複数の項を追加する．

  addTerms メソッドの引数の意味は以下の通り．
   -    coeffs は追加する項の係数もしくは係数リスト．係数もしくはリストの要素は整数．
   -    vars は追加する項の変数インスタンスもしくはの変数インスタンスのリスト．リストの場合には，リストcoeffsと同じ長さをもつ必要がある．
   -    values は追加する項の値もしは値のリスト．リストの場合には，リストcoeffsと同じ長さをもつ必要がある．


  addTerms メソッドは，1つの項を追加するか，複数の項を一度に追加する．1つの項を追加する場合には，引数の係数は整数値，変数は変数インスタンスで与え，値は変数の領域の要素とする．複数の項を一度に追加する場合には，同じ長さをもつ，係数，変数インスタンス，値のリストで与える．

たとえば，項をもたない線形制約インスタンス L に対して，
```
L.addTerms(1, y, 'A')
```
と1つの項を追加すると，制約の左辺は
```
1 x[y, 'A']
```
となる．ここで  x は値変数（ y が 'A' になるとき 1，それ以外のとき 0 の仮想の変数）を表す．

同様に，項をもたない線形制約インスタンス  L に対して，
```
L.addTerms([2, 3, 1], [y, y, z], ['C', 'D', 'C'])
```
と3つの項を同時に追加すると，制約の左辺は以下のようになる．
```
 2 x[y,'C']  + 3 x[y,'D'] + 1 x[z,'C']
```
-    setRhs(rhs) は線形制約の右辺定数を  rhs に設定する．引数は整数値であり，既定値は 0 である．

-    setDirection(dir) は制約の向きを設定する．引数  dir は
  '<=' ,   '>=' ,   '=' のいずれかの文字列とする．既定値は  '<=' である．

-    setWeight(weight) は制約の重みを  weight に設定する．引数は正数値もしくは文字列  'inf' である．
ここで  'inf' は無限大を表し，絶対制約を定義するときに用いられる．

また，線形制約クラス  Linear は，制約の情報を文字列として返すことができる．

```{.python.marimo disabled="true" hide_code="true"}
#| export
class Linear(Constraint):
    """
    Linear ( name, weight=1, rhs=0, direction="<=" )
    Linear constraint constructor.

    Arguments:
    - name: Name of linear constraint.
    - weight (optiona): Positive integer representing importance of constraint.
    - rhs: Right-hand-side constant of linear constraint.
    - direction: Rirection (or sense) of linear constraint; "<=" (default) or ">=" or "=".

    Attributes:
    - name: Name of linear constraint.
    - weight (optional): Positive integer representing importance of constraint.
    - rhs: Right-hand-side constant of linear constraint.
    - lhs: Left-hand-side constant of linear constraint.
    - direction: Direction (or sense) of linear constraint; "<=" (default) or ">=" or "=".
    - terms: List of terms in left-hand-side of constraint. Each term is a tuple of coeffcient,variable and its value.
    """
    name: str                 = ""
    weight: Optional[Union[int,str]]     = 1
    rhs: Optional[int]        = 0
    direction: Optional[str]  = "<="
    terms: Optional[List[Tuple[int,Variable,str]]] = []
    lhs: int                  = 0

    def __init__(self, name="", weight=1, rhs=0, direction="<="):
        """
        Constructor of linear constraint class:
        """
        super(Linear, self).__init__(name = name, weight = weight)

        if direction in ["<=", ">=", "="]:
            self.direction = direction
        else:
            raise NameError("direction setting error;direction should be one of '<=', '>=', or '='")


    def __str__(self):
        """ 
            return the information of the linear constraint
            the constraint is expanded and is shown in a readable format
        """
        f =["{0}: weight= {1} type=linear".format(self.name, self.weight)]
        for (coeff,var,value) in self.terms:
            f.append( "{0}({1},{2})".format(str(coeff),var.name,str(value)) )
        f.append( self.direction+str(self.rhs) +"\n" )
        return " ".join(f)

    def addTerms(self,coeffs=[],vars=[],values=[]):
        """
            - addTerms ( coeffs=[],vars=[],values=[] )
            Add new terms into left-hand-side of linear constraint.

            Arguments:
            - coeffs: Coefficients for new terms; either a list of coefficients or a single coefficient. The three arguments must have the same size.
            - vars: Variables for new terms; either a list of variables or a single variable. The three arguments must have the same size.
            - values: Values for new terms; either a list of values or a single value. The three arguments must have the same size.

            Example usage:

            L.addTerms(1, y, "A")
            L.addTerms([2, 3, 1], [y, y, z], ["C", "D", "C"]) #2 X[y,"C"]+3 X[y,"D"]+1 X[z,"C"]
        """
        if type(coeffs) !=type([]): #need a check whether coeffs is numeric ...
            #arguments are not a list; add a term
            if type(coeffs)==type(1):  #整数の場合だけ追加する．
                self.terms.append( (coeffs,vars,str(values)))
            else:
                raise ValueError("Coefficient must be an integer.")
        elif type(coeffs)!=type([]) or type(vars)!=type([]) or type(values)!=type([]):
            raise TypeError("coeffs, vars, values must be lists")
        elif len(coeffs)!=len(vars) or len(coeffs)!=len(values):
            raise TypeError("length of coeffs, vars, values must be identical")
        else:
            for i in range(len(coeffs)):
                self.terms.append( (coeffs[i],vars[i],str(values[i])))

    def setRhs(self,rhs=0):
        if type(rhs) != type(1):
            raise ValueError("Right-hand-side must be an integer.")
        else:
            self.rhs = rhs

    def setDirection(self,direction="<="):
        if direction in ["<=",">=","="]:
            self.direction = direction
        else:
            raise NameError(
                "direction setting error; direction should be one of '<=', '>=', or '='"
                           )

    def feasible(self,allvars):
        """ 
        return True if the constraint is defined correctly
        """
        for (coeff,var,value) in self.terms:
            if var.name not in allvars:
                raise NameError("no variable in the problem instance named %r" % var.name)
            if value not in allvars[var.name].domain:
                raise NameError("no value %r for the variable named %r" % (value, var.name))
        return True
```

### Linearクラスの使用例

```python
_L = Linear(name='a linear constraint', weight='inf', rhs=10, direction='<=')
_x = Variable(name='x', domain=['A', 'B', 'C'])
_L.addTerms(3, _x, 'A')
print(_L)

try:
    _L = Linear(name='a linear constraint', rhs=10.56, direction='=')
except ValueError as error:
    print(error)
_x = Variable(name='x', domain=['A', 'B', 'C'])
try:
    _L.addTerms(3.1415, _x, 'A')
except ValueError as error:
    print(error)
```

出力
```python
a_linear_constraint: weight= inf type=linear 3(x,A) <=0

Coefficient must be an integer.
```
<!---->
## 2次制約クラス Quadraric

SCOPでは（非凸の）2次関数を左辺にもつ制約（2次制約）も扱うことができる．

2次制約は，

$$
  2次項1 + 2次項2 + \cdots  制約の方向 (\leq, \geq, =) 右辺定数
$$

の形で与えられる．ここで2次項は，

$$
 係数 \times x[変数1,値1] \times  x[変数2,値2]
$$

で与えられる．


2次制約クラス  Quadratic のインスタンスは，以下のように生成する．

```python
2次制約インスタンス=Quadratic(name, weight=1, rhs=0, direction='<=')
```

2次制約クラスの引数と属性は，線形制約クラス  Linear と同じである．

Quadratic クラスは，2次制約の左辺に2つの変数の積から成る項を追加する以下のメソッドをもつ．

```python
addTerms(coeffs,vars1,values1,vars2,values2)
```

2次制約に対する addTerms メソッドの引数は以下の通り．

-    coeffs は追加する項の係数もしくは係数のリスト．係数もしくはリストの要素は整数．
-    vars1 は追加する項の第1変数インスタンスもしくは変数インスタンスのリスト． リストの場合には，リスト  coeffs と同じ長さをもつ必要がある．
-    values1 は追加する項の第1変数の値もしくは値のリスト． リストの場合には，リスト coeffs と同じ長さをもつ必要がある．
-    vars2 は追加する項の第2変数の変数インスタンスもしくは変数インスタンスのリスト． リストの場合には，リスト  coeffs と同じ長さをもつ必要がある．
-    values2 は追加する項の第2変数の値もしくは値のリスト． リストの場合には， リスト coeffs と同じ長さをもつ必要がある．

addTerms メソッドは，1つの項を追加するか，複数の項を一度に追加する．
1つの項を追加する場合には，
引数の係数は整数値，変数は変数インスタンスで与え，値は変数の領域の要素とする．
複数の項を一度に追加する場合には，同じ長さをもつ，係数，変数インスタンス，値のリストで与える．

たとえば，項をもたない2次制約インスタンス  Q に対して，
```python
Q.addTerms(1, y, 'A', z, 'B')
```
と1つの項を追加すると，制約の左辺は
```python
1 x[y,'A'] * x[z,'B']
```
となる．

同様に，項をもたない2次制約インスタンス  Q に対して，
```python
Q.addTerms([2, 3, 1], [y, y, z], ['C', 'D', 'C'], [x, x, y], ['A', 'B', 'C'])
```
と3つの項を同時に追加すると，制約の左辺は以下のようになる．
```python
2 x[y,'C'] * x[x,'A'] + 3 x[y,'D'] * x[x,'B'] + 1 x[z,'C'] * x[y,'C']
```

-   setRhs(rhs) は2次制約の右辺定数を  rhs に設定する．引数は整数値であり，既定値は 0 である．

-    setDirection(dir) は制約の向きを設定する．引数  dir は
  '<=' ,   '>=' ,   '=' のいずれかの文字列とする．既定値は  '<=' である．

-    setWeight(weight) は制約の重みを設定する．引数は正数値もしくは文字列  'inf' である．
  'inf' は無限大を表し，絶対制約を定義するときに用いられる．

また，2次制約クラス  Quadratic は，制約の情報を文字列として返すことができる．

```{.python.marimo disabled="true" hide_code="true"}
#| export
class Quadratic(Constraint):
    """
    Quadratic ( name, weight=1, rhs=0, direction="<=" )
    Quadratic constraint constructor.

    Arguments:
    - name: Name of quadratic constraint.
    - weight (optional): Positive integer representing importance of constraint.
    - rhs: Right-hand-side constant of linear constraint.
    - direction: Direction (or sense) of linear constraint; "<=" (default) or ">=" or "=".

    Attributes:
    - name: Name of quadratic constraint.
    - weight (optiona): Positive integer representing importance of constraint.
    - rhs: Right-hand-side constant of linear constraint.
    - lhs: Left-hand-side constant of linear constraint.
    - direction: Direction (or sense) of linear constraint; "<=" (default) or ">=" or "=".
    - terms: List of terms in left-hand-side of constraint. Each term is a tuple of coeffcient, variable1, value1, variable2 and value2.
    """

    name: str                 = ""
    weight: Optional[Union[int,str]]     = 1
    rhs: Optional[int]        = 0
    direction: Optional[str]  = "<="
    terms: Optional[List[Tuple[int,Variable,str,Variable,str]]] = []
    lhs: int                  = 0

    def __init__(self, name="", weight=1, rhs=0, direction="<="):
        super(Quadratic,self).__init__(name=name, weight=weight)

        if direction in ["<=", ">=", "="]:
            self.direction = direction
        else:
            raise NameError(
                "direction setting error;direction should be one of '<=', '>=', or '='"
                  )


    def __str__(self):
        """ return the information of the quadratic constraint
            the constraint is expanded and is shown in a readable format
        """
        f = [ "{0}: weight={1} type=quadratic".format(self.name,self.weight) ]
        for (coeff,var1,value1,var2,value2) in self.terms:
            f.append( "{0}({1},{2})({3},{4})".format(
                str(coeff),var1.name,str(value1),var2.name,str(value2)
                ))
        f.append( self.direction+str(self.rhs) +"\n" )
        return " ".join(f)

    def addTerms(self,coeffs=[],vars=[],values=[],vars2=[],values2=[]):
        """
        addTerms ( coeffs=[],vars=[],values=[],vars2=[],values2=[])

        Add new terms into left-hand-side of qua
        dratic constraint.

        Arguments:
        - coeffs: Coefficients for new terms; either a list of coefficients or a single coefficient. The five arguments must have the same size.
        - vars: Variables for new terms; either a list of variables or a single variable. The five arguments must have the same size.
        - values: Values for new terms; either a list of values or a single value. The five arguments must have the same size.
        - vars2: Variables for new terms; either a list of variables or a single variable. The five arguments must have the same size.
        - values2: Values for new terms; either a list of values or a single value. The five arguments must have the same size.

        Example usage:

        L.addTerms(1, y, "A", z, "B")

        L.addTerms([2, 3, 1], [y, y, z], ["C", "D", "C"], [x, x, y], ["A", "B", "C"])
                  #2 X[y,"C"] X[x,"A"]+3 X[y,"D"] X[x,"B"]+1 X[z,"C"] X[y,"C"]

        """
        if type(coeffs) !=type([]): 
            if type(coeffs)==type(1):  #整数の場合だけ追加する．
                self.terms.append( (coeffs,vars,str(values),vars2,str(values2)))
            else:
                raise ValueError("Coefficient must be an integer.")
        elif type(coeffs)!=type([]) or type(vars)!=type([]) or type(values)!=type([]) \
             or type(vars2)!=type([]) or type(values2)!=type([]):
            raise TypeError("coeffs, vars, values must be lists")
        elif len(coeffs)!=len(vars) or len(coeffs)!=len(values) or len(values)!=len(vars) \
             or len(coeffs)!=len(vars2) or len(coeffs)!=len(values2):
            raise TypeError("length of coeffs, vars, values must be identical")
        else:
            for i in range(len(coeffs)):
                self.terms.append( (coeffs[i],vars[i],str(values[i]),vars2[i],str(values2[i])))

    def setRhs(self,rhs=0):
        if type(rhs) != type(1):
            raise ValueError("Right-hand-side must be an integer.")
        else:
            self.rhs = rhs

    def setDirection(self,direction="<="):
        if direction in ["<=", ">=", "="]:
            self.direction = direction
        else:
            raise NameError(
                "direction setting error;direction should be one of '<=', '>=', or '='"
                  )

    def feasible(self,allvars):
        """
          return True if the constraint is defined correctly
        """
        for (coeff,var1,value1,var2,value2) in self.terms:
            if var1.name not in allvars:
                raise NameError("no variable in the problem instance named %r" % var1.name)
            if var2.name not in allvars:
                raise NameError("no variable in the problem instance named %r" % var2.name)
            if value1 not in allvars[var1.name].domain:
                raise NameError("no value %r for the variable named %r" % (value1, var1.name))
            if value2 not in allvars[var2.name].domain:
                raise NameError("no value %r for the variable named %r" % (value2, var2.name))
        return True
```

### Quadraticクラスの使用例

```python
_q = Quadratic(name='a quadratic constraint', rhs=10, direction='<=')
_x = Variable(name='x', domain=['A', 'B', 'C'])
_y = Variable(name='y', domain=['A', 'B', 'C'])
_q.addTerms([3, 9], [_x, _x], ['A', 'B'], [_y, _y], ['B', 'C'])
print(_q)
```

出力
```python
a_quadratic_constraint: weight=1 type=quadratic 3(x,A)(y,B) 9(x,B)(y,C) <=0
```
<!---->
## 相異制約クラス Alldiff

相異制約は，変数の集合に対し, 集合に含まれる変数すべてが異なる値を
とらなくてはならないことを規定する．
これは組合せ的な構造に対する制約であり，制約最適化の特徴的な制約である．

SCOPにおいては，値が同一であるかどうかは，値の名称ではなく，
変数のとりえる値の集合（領域）を表したリストにおける**順番（インデックス）**によって決定される.
たとえば, 変数  var1 および  var2 の領域がそれぞれ  ['A','B'] ならびに  ['B','A'] であったとき，
変数  var1 の値   'A', 'B'  の順番はそれぞれ 0 と 1，
変数  var2 の値   'A', 'B'  の順番はそれぞれ 1 と 0 となる．
したがって，**相異制約を用いる際には変数に同じ領域を与える**ことが（混乱を避けるという意味で）推奨される．

相異制約クラス  Alldiff のインスタンスは，以下のように生成する．

```python
相異制約インスタンス = Alldiff(name, varlist, weight)
```

引数の名前と既定値は以下の通り．


-    name は制約名を与える．

-    varlist は相異制約に含まれる変数インスタンスのリストを与える．
これは，値の順番が異なることを要求される変数のリストであり，省略も可能である．
その場合の既定値は，空のリストとなる．
ここで追加する変数は，モデルクラスに追加された変数である必要がある．

-    weight は制約の重みを与える．

相異制約の制約名と重みについては，線形制約クラス  Linear と同じように設定する．
上の引数は  Alldiff クラスの属性でもある．その他の属性として最適化した後で得られる式の評価値がある．


-    lhs は左辺(left hand side)の評価値を表し，最適化された後に，同じ値の番号（インデックス）をもつ変数の数が代入される．


  Alldiff クラスは，以下のメソッドをもつ．


-    addVariable(var) は相異制約に1つの変数インスタンス  var を追加する．

-    addVariables(varlist) は相異制約の変数インスタンスを複数同時に（リスト  varlist として）追加する．

-    setWeight(weight) は制約の重みを設定する．引数は正数値もしくは文字列  'inf' である．  'inf' は無限大を表し，絶対制約を定義するときに用いられる．


また，相異制約クラス  Alldiff は，制約の情報を文字列として返すことができる．

```{.python.marimo disabled="true" hide_code="true"}
#| export
class Alldiff(Constraint):
    """
    Alldiff ( name=None,varlist=None,weight=1 )
    Alldiff type constraint constructor.

    Arguments:
    - name: Name of all-different type constraint.
    - varlist (optional): List of variables that must have differennt value indices.
    - weight (optional): Positive integer representing importance of constraint.

    Attributes:
    - name: Name of all-different type  constraint.
    - varlist (optional): List of variables that must have differennt value indices.
    - lhs: Left-hand-side constant of linear constraint.

    - weight (optional): Positive integer representing importance of constraint.
    """

    name: str                 = ""
    weight: Optional[Union[int,str]]     = 1
    varlist: List[Variable]   = []
    lhs: int                  = 0   

    def __init__(self, name="", varlist=None, weight=1):
        super(Alldiff,self).__init__(name=name, weight=weight)
        self.lhs=0
        if varlist==None:
            self.varlist = []
        else:
            for var in varlist:
                if not isinstance(var, Variable):
                    raise NameError("error: %r should be a subclass of Variable" % var)
            self.varlist = varlist

    def __str__(self):
        """
        return the information of the alldiff constraint
        """
        f = [ "{0}: weight= {1} type=alldiff ".format(self.name,self.weight) ]
        for var in self.varlist:
            f.append( var.name )
        f.append( "; \n" )
        return " ".join(f)

    def addVariable(self,var):
        """
        addVariable ( var )
        Add new variable into all-different type constraint.

        Arguments:
        - var: Variable object added to all-different type constraint.

        Example usage:

        AD.addVaeiable( x )

        """
        if not isinstance(var, Variable):
            raise NameError("error: %r should be a subclass of Variable" % var)

        if var in self.varlist:
            print("duplicate variable name error when adding variable %r" % var)
            return False
        self.varlist.append(var)

    def addVariables(self, varlist):
        """
        addVariables ( varlist )
        Add variables into all-different type constraint.

        Arguments:
        - varlist: List or tuple of variable objects added to all-different type constraint.

        Example usage:

        AD.addVariables( x, y, z )

        AD.addVariables( [x1,x2,x2] )

        """
        for var in varlist:
            self.addVariable(var)

    def feasible(self, allvars):
        """
           return True if the constraint is defined correctly
        """
        for var in self.varlist:
            if var.name not in allvars:
                raise NameError("no variable in the problem instance named %r" % var.name)
        return True
```

### Alldiffクラスの使用例

```python
_A = Alldiff(name='a alldiff constraint')
_x = Variable(name='x', domain=['A', 'B', 'C'])
_y = Variable(name='y', domain=['A', 'B', 'C'])
_A.addVariables([_x, _y])
print(_A)
```

出力
```python
a_alldiff_constraint: weight= 1 type=alldiff  x y ;
```
<!---->
## 最適化の描画関数 plot_scop

SCOPはメタヒューリスティクスによって解の探索を行う． 一般には，解の良さと計算時間はトレードオフ関係がある．つまり，計算時間をかければかけるほど良い解を得られる可能性が高まる．
どの程度の計算時間をかければ良いかは，最適化したい問題例（問題に数値を入れたもの）による． plot_scopは，横軸に計算時間，縦軸に目的関数値をプロットする関数であり，最適化を行ったあとに呼び出す．
得られるPlotlyの図は，どのくらいの計算時間をかければ良いかをユーザーが決めるための目安を与える．

たとえば以下の例の図から，500秒程度の計算時間で良好な解を得ることができるが，念入りにさらに良い解を探索したい場合には2000秒以上の計算時間が必要なことが分かる．

```{.python.marimo disabled="true" hide_code="true"}
#| export
def plot_scop(file_name: str="scop_out.txt"):
    with open(file_name) as f:
        out = f.readlines()
    x, y1, y2 = [],[],[] 
    for l in out[5:]: 
        sep = re.split("[=()/]", l)
        #print(sep)
        if sep[0] == '# penalty ':
            break
        if sep[0] == 'penalty ':
            hard, soft, cpu = map(float, [ sep[1], sep[2], sep[6]])
            x.append(cpu)
            y1.append(hard)
            y2.append(soft)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x = x, 
            y = y1,
            mode='markers+lines',
            name= "hard",
            marker=dict(
                size=10,
                color= "red")
    ))
    fig.add_trace(go.Scatter(
            x = x, 
            y = y2,
            name ="soft",
            mode='markers+lines',
            marker=dict(
                size=8,
                color= "blue")
    ))
    fig.update_layout(title = "SCOP performance",
                   xaxis_title='CPU time',
                   yaxis_title='Penalty')
    return fig
```

## 最適化の描画関数 plot_scop の実行例


``` python
fig = plot_scop()
plotly.offline.plot(fig);
```

<img src="https://github.com/scmopt/scmopt_data/blob/main/fig/plot-scop.png?raw=true" width=600 height=200>
<!---->
## SCOPのデバッグのコツ

ここでは，SCOPのデバッグのコツについて述べる．

SCOPでは，変数や制約やモデル全体のインスタンスをprint関数で表示することができる． プログラムを少しずつ書き， printでインスタンスが意図通りになっているかを出力して確認する．

SCOPは，Pythonからテキストファイルを生成して，それを実行ファイルに入れて求解する．入力ファイル名は **scop_input.txt** で出力ファイル名は **scop_output.txt** である．
これらのファイルをエディタでみて，意図通りかどうかを確認することもできる．

その際に，大きな問題例だとデバッグしにくいので，非常に小さい例題を準備することを推奨する． また，制約条件も簡単なものから順に複雑な制約に変更していき，
実行不能になったときの原因が容易に分かるようにすべきである．
<!---->
## 大規模な実際問題の解決法

- 大きな問題例(問題に数値を与えたもの；instance)を直接モデル化して実験するのではなく，小規模ですべての機能をテストできるような問題例を準備すべき．

- SCOPでは，データはすべて整数値にしなければならない．

- 1つの式の評価が大きな整数になることを避ける．

- 大きな整数になる可能性がある項の和に対する線形制約のかわりに，項を分解してそれぞれの式を評価する．

- 以前探索を行ったときの最良解の情報は，「変数:値」を1行としたテキストとしてファイル名 scop_best_data.txt に保管されている．
これを初期解として再最適化を行うには， model.Params.InitialをTrueに設定する．これによって2回め以降の探索の高速化が可能になる．
また，このファイルを書き換えることによって，異なる初期解から探索を行うこともできる．
<!---->
## ターミナル（コマンドプロンプト）での実行


SCOPは，Pythonからテキストファイルを生成して，それを実行ファイルに入れて求解する．入力ファイル名は scop_input.txt で出力ファイル名は scop_output.txt である．
デバッグ時には，入力ならびに出力ファイルを適当なテキストエディタで開いてみると良い．

また，デバッグ時だけでなく， 大規模問題例を解くときや，アルゴリズムのログを確認しながら実験したい場合には，ターミナル（コマンドプロンプト）からの実行を推奨する．

コマンドはOSによって異なるが，scop(Mac), scop-linux (linux), scop.exe (Windows) を以下の形式で実行する．

```
./scop       < scop_input.txt  #Mac

./scop-linux < scop_input.txt  #linux

scop         < scop_input.txt  #Windows
```

オプションの引数は， 「-オプション名 引数」の形式で追加する． scop --help で表示されるオプションは，以下の通りである．

```
  -noadjust         deactivate weight adjustment mechanism
  -display #        set log display level
  -iteration #      set iteration limit
  -initsolfile file set initial solution file name
  -inputfile file   set input file name
  -interval #       set log display interval
  -logfile file     set log file name
  -noimprovement #  set iteration limit for no improvement
  -seed #           set random seed
  -logfile file     set log file name
  -target #         set target
  -time #.#         set cpu time limit in second
```

たとえば，実行時のログを1反復ごとに表示させたい場合には，

```
./scop     -interval 1 < scop_input.txt

./scop-linux -interval 1 < scop_input.txt

scop         -interval 1 < scop_input.txt
```

とすれば良い．
<!---->
## 例題

ここでは，幾つかの簡単な例題を通してSCOPの基本的な使用法を解説する．

以下の例題を動かすためには，最初に以下を追加する必要がある．

```python
from scop import *
```

もしくは（Marimoのように * でのインポートが禁止されている場合）

```python
from scop import Model, Variable, Linear, Quadratic, Alldiff
```
<!---->
### 仕事の割当1

あなたは，土木事務所の親方だ．いま，3人の作業員 A,B,C を3つの仕事 $0,1,2$ に割り当てる必要がある．
すべての仕事には1人の作業員を割り当てる必要があるが，
作業員と仕事には相性があり，割り当てにかかる費用（単位は万円）は，以下のようになっているものとする．

|  |      |     |     |
| ---- | ---- |---- |---- |
| 仕事  |  0  |  1  |  2    |
| 作業員 |     |     |      |
|A       |  15 |  20  |  30  |
|B       |  7  |  15  |  12   |
|C       |  25  |  10  |  13  |

総費用を最小にするように作業員に仕事を割り振るには，どのようにしたら良いだろうか？

この問題をSCOPを使って解いてみよう．

まず，モデルのインスタンス`model`を生成し，作業員 $A,B,C$ に割り振られた仕事を表す変数 $X_A, X_B, X_C$ （プログラムでは `A,B,C`）を定義する．
数理最適化においては変数は数字で表さなければならないが， 制約最適化では，変数のとれる値の集合で定義する．
これを **領域** (domain)とよぶ． 作業員は仕事 $0,1,2$ のいずれかの仕事をすることができるので，各変数の領域は`[0,1,2]`となる．
変数の追加は，モデルインスタンス`model`の`addVariable`メソッドを用いる．

```python
model = Model()
A = model.addVariable(name="A", domain=[0,1,2])
B = model.addVariable(name="B", domain=[0,1,2])
C = model.addVariable(name="C", domain=[0,1,2])
```

数理最適化でモデリングをすると，3人の作業員に対して3つの仕事に割り当てるか否かを表す 9個の0-1変数が必要になるが， SCOPだと3個の変数で表現できる．

すべての仕事に1人の作業員を割り当てることを表すには， 相異制約を使う．

- 相異制約 (Alldiff): リストに含まれる変数（すべて同じ領域をもつと仮定する）がすべて異なる値をとることを表す．

`Alldiff`クラスのインスタンス`alldiff`を生成し， それを`model`に追加する．
SCOPにおける制約は， すべて逸脱したときのペナルティを`weight`引数で定義する．`weight`引数に無限大を表す`inf`を入れると，絶対制約（ハードな制約）を定義できる．

```python
alldiff = Alldiff("All Diff",[A,B,C],weight="inf")
model.addConstraint(alldiff)
```

これも数理最適化でモデリングすると，仕事ごとに定義する必要があるので 3本の制約が必要であるが， SCOPだと相異制約1本で表現できる．

SCOPには目的関数という概念がない． すべて制約で表現し，制約の逸脱ペナルティの合計を最小化する． 割り当て費用は線形制約で記述する．

線形制約は線形不等式（もしくは等式）であり，式として記述する際には，値変数の概念を用いる． 値変数とは変数が領域の値をとったときに $1$ になる仮想の変数であり，
実際のプログラム内では使わない． 作業員 $A$ に割り当てられた仕事を表す変数 $X_A$ に対して，値変数 $x_{Aj} (j=0,1,2)$ が定義される．
$x_{Aj}$ は， 作業員 $A$ が仕事 $j$ に割り当てられたときに $1$，それ以外のとき $0$ を表す変数である．

これを使うと割り当て費用を表す線形制約は，

$$
15 x_{A0} + 20 x_{A1} + 30 x_{A2} + 7 x_{B0} + 15 x_{B1} + 12 x_{B2} + 25 x_{C0} + 10 x_{C1} + 13 x_{C2} \leq 0
$$

と書ける． この制約の逸脱ペナルティ`weight`を $1$ に設定すると，制約の逸脱を許す考慮制約（ソフトな制約）となり，逸脱量が割り当て費用になる．

線形制約クラス`Linear`の右辺定数`rhs`を $0$，制約の方向を`<=`と設定してインスタンス`linear`を生成する．
左辺の各項は，`addTerms`メソッドを用いて追加する．引数は順に，係数のリスト，変数のリスト，値のリストである．

```python
linear = Linear(name="Objective Function",weight=1,rhs=0,direction="<=")
linear.addTerms([15,20,30],[A,A,A],[0,1,2])
linear.addTerms([7,15,12],[B,B,B],[0,1,2])
linear.addTerms([25,10,13],[C,C,C],[0,1,2])
model.addConstraint(linear)
```

SCOPの変数，制約，モデルなどの各インスタンスは`print`関数で表示できる． ここでは上で作成したモデルインスタンス`model`を表示しておく．
`model`の`optimize`メソッドで最適化を実行する．返値は解を表す辞書と逸脱した制約を表す辞書である．

```python
print(model)
sol, violated = model.optimize()
print("solution=", sol)
print("violated constraint=", violated)
```

結果は以下のように表示される．

```
Model:
number of variables = 3
number of constraints= 2
variable A:['0', '1', '2'] = None
variable B:['0', '1', '2'] = None
variable C:['0', '1', '2'] = None
AD: weight= inf type=alldiff  C A B ;  :LHS =0
linear_constraint: weight= 1 type=linear 15(A,0) 20(A,1) 30(A,2) 7(B,0) 15(B,1) 12(B,2) 25(C,0) 10(C,1) 13(C,2) <=0 :LHS =0

 ================ Now solving the problem ================

solution
A 0
B 2
C 1
violated constraint(s)
linear_constraint 37
```
<!---->
### 仕事の割当2

あなたは土木事務所の親方だ．今度は，5人の作業員 A,B,C,D,Eを3つの仕事 $0,1,2$ に割り当てる必要がある．
ただし，各仕事にかかる作業員の最低人数が与えられており，それぞれ $1,2,2$人必要であり，
割り当ての際の費用（単位は万円）は，以下のようになっているものとする．

|      |      |     |     |
| ---- | ---- |---- |---- |
| 仕事  |  0  |  1  |  2    |
| 作業員 |      |     |     |
|A       |  15 |  20  |  30 |
|B       |  7  |  15  |  12 |
|C       |  25  |  10  |  13  |
|D      |  15  |  18 |   3  |
|E       |  5  |  12  |  17  |

さて，誰にどの仕事を割り振れば費用が最小になるだろうか？


例題1では変数を`A,B,C`と別々に定義したが，ここではより一般的な記述法を示す．

パラメータは例題1と同じようにリストと辞書で準備する．

- $W$，: 作業員の集合． その要素を $i$ とする．
- $J$:  仕事の集合． その要素を $j$ とする．
- $c_{ij}$: 作業員 $i$ が仕事 $j$ に割り当てられたときの費用
- $LB_j$: 仕事 $j$ に必要な人数

```python
model=Model()
workers=['A','B','C','D','E']
Jobs   =[0,1,2]
Cost={ ('A',0):15, ('A',1):20, ('A',2):30,
       ('B',0): 7, ('B',1):15, ('B',2):12,
       ('C',0):25, ('C',1):10, ('C',2):13,
       ('D',0):15, ('D',1):18, ('D',2): 3,
       ('E',0): 5, ('E',1):12, ('E',2):17
       }
LB={0: 1,
    1: 2,
    2: 2
    }
```

変数は辞書`x`に保管する．

- $X_{i}$: 作業員 $i$ に割り振られた仕事を表す変数． 領域は仕事の集合 $J$ であり，そのうち1つの「値」を選択する．

```python
x={}
for i in workers:
    x[i]=model.addVariable(name=i,domain=Jobs)
```

$x_{ij}$ は， $X_i$ が $j$ に割り当てられたときに $1$，それ以外のとき $0$ を表す変数（値変数）であり，これを使うと人数の下限制約と割り当て費用は，以下の
線形制約として記述できる．

人数下限を表す線形制約（重み $\infty$）

$$
\sum_{i \in W} x_{ij} \geq LB_j  \ \ \ \forall j \in J
$$


割り当て費用を表す線形制約（重み $1$）

$$
\sum_{i \in W, j \in J} c_{ij} x_{ij} \leq 0
$$

```python
LBC={}
for j in Jobs:
    LBC[j]=Linear(f"LB{j}","inf",LB[j],">=")
    for i in workers:
        LBC[j].addTerms(1,x[i],j)
    model.addConstraint(LBC[j])

obj=Linear("obj")
for i in workers:
    for j in [0,1,2]:
        obj.addTerms(Cost[i,j],x[i],j)
model.addConstraint(obj)
```

```python
`model`のパラメータ`Params`で制限時間`TimeLimit`を1（秒）に設定して最適化する．

model.Params.TimeLimit=1
sol,violated=model.optimize()

print('solution')
for x in sol:
    print (x,sol[x])
print ('violated constraint(s)')
for v in violated:
    print (v,violated[v])
```


結果は以下のように表示される．

```
solution
A 1
B 2
C 1
D 2
E 0
violated constraint(s)
obj 50
```
<!---->
### 仕事の割当3

上の例題と同じ状況で，仕事を割り振ろうとしたところ，作業員 A と C は仲が悪く，
一緒に仕事をさせると喧嘩を始めることが判明した．
作業員 A と C を同じ仕事に割り振らないようにするには，どうしたら良いだろうか？

この問題は，追加された作業員 A と C を同じ仕事に割り当てることを禁止する制約を記述するだけで解決できる．
ここでは，2次制約（重みは $100$）として記述する．

$$
x_{A0} x_{C0} + x_{A1} x_{C1} + x_{A2} x_{C2} = 0
$$

作業員AとCが同じ仕事に割り当てられると左辺は $1$になり，制約を逸脱する．

線形制約クラスと同様に2次制約クラス`Quadratic`からインスタンス`conf`を生成する．
左辺の項を追加するには，`addTerms`メソッドを用いる． 引数は，最初の変数の係数，変数，値の次に2番目の変数の係数，変数，値を入れる．

```python
conf=Quadratic("conflict",100,0,"=")
for j in Jobs:
    conf.addTerms(1,x["A"],j,x["C"],j)
model.addConstraint(conf)
```

数理最適化ソルバーは非凸の2次を含む制約や目的関数が苦手であるが，SCOPは通常の制約と同じように解くことができる．

結果は以下のように表示される．

```
solution
A 0
B 2
C 1
D 2
E 1
violated constraint(s)
obj 52
```
<!---->
### 魔方陣

魔方陣とは， $n \times n$ のマス目に $1$ から $n^2$ までの数字を1つずつ入れて，どの横行，縦列，対角線のマス目の数字の和も同じになるようにしたものである.

$n=3$ の問題を以下の手順で問題を解く．

1. 各マス目 $(i,j), i=0,1,2, j=0,1,2$ に対して変数 $x[i,j]$ を準備して，その領域を $1$ から $9$ までの数とする．

2. 各マス目には異なる数字を入れる必要があるので，すべての変数のリストを入れた相異制約 (Alldiff) を追加する． この制約は絶対制約とする．

3. さらに，各行（$i=0,1,2$)と各列($j=0,1,2$)に対して，その和がちょうど $15 = (1+2+\cdots+9)/3$ になるという制約を追加する． これらの制約は考慮制約とし，逸脱の重みは $1$ とする．

4. 最適化を行い，解を表示する．


- 行の集合を $I$， その要素を $i$ とする．

- 列の集合を $J$，その要素を $j$ とする．

- $X_{ij}$: マス目 $i,j$ に割り当てられた数字を表す変数； 領域は $[1,2,3,4,5,6,7,8,9]$ であり，そのうち1つの「値」を選択する．

- $x_{ijk}$: $X_{ij}$ が $k$ に割り当てられたときに $1$，それ以外のとき $0$ を表す変数（値変数）


相異制約（重み $\infty$）； すべてのマス目の数字が異なることを表す．

```
Alldiff( [ X_{ij} for i in I for j in J  ] )
```

線形制約（重み $1$）；行ごとの和が $15$ であることを表す．

$$
\sum_{j \in J} \sum_{k} k x_{ijk} = 15 \ \ \ \forall i \in I
$$

線形制約（重み $1$）；列ごとの和が $15$ であることを表す．

$$
\sum_{i \in I} \sum_{k} k x_{ijk} = 15 \ \ \ \forall j \in J
$$

線形制約（重み $1$）；対角線ごとの和が $15$ であることを表す．

$$
\sum_{j \in J} \sum_{k} k x_{jjk} = 15
$$

$$
\sum_{j \in J} \sum_{k} k x_{j,2-j,k} = 15
$$

以下に一般の $n$ でも解けるプログラムを示す． ただしトライアル版だと $n=3$ までしか解くことができない．

```python
n = 3
nn = n*n
model = Model()
x = {}
dom = [i+1 for i in range(nn)]
sum_ = sum(dom)//n
for i in range(n):
    for j in range(n):
        x[i,j] = model.addVariable(name=f"x[{i},{j}]", domain=dom)
alldiff = Alldiff(f'AD',[ x[i,j] for i in range(n) for j in range(n) ], weight='inf')
model.addConstraint( alldiff )
col_constr = {}
for j in range(n):
    col_constr[j] =  Linear(f'col_constraint{j}',weight=1,rhs=sum_,direction='=')
    for i in range(n):
        for k in range(1,nn+1):
            col_constr[j].addTerms(k,x[i,j],k)
    model.addConstraint(col_constr[j])
row_constr = {}
for i in range(n):
    row_constr[i] =  Linear(f'row_constraint{i}',weight=1,rhs=sum_,direction='=')
    for j in range(n):
        for k in range(1,nn+1):
            row_constr[i].addTerms(k,x[i,j],k)
    model.addConstraint(row_constr[i])
diagonal_constr = {}
diagonal_constr[0] =  Linear(f'diagonal_constraint{0}',weight=1,rhs=sum_,direction='=')
for j in range(n):
    for k in range(1,nn+1):
        diagonal_constr[0].addTerms(k,x[j,j],k)
model.addConstraint(diagonal_constr[0])
diagonal_constr[1] =  Linear(f'diagonal_constraint{1}',weight=1,rhs=sum_,direction='=')
for j in range(n):
    for k in range(1,nn+1):
        diagonal_constr[1].addTerms(k,x[j,n-1-j],k)
model.addConstraint(diagonal_constr[1])
model.Params.TimeLimit=100
model.Params.RandomSeed=1
#model.Params.OutputFlag=True
sol,violated = model.optimize()
print("逸脱制約=", violated)
import numpy as np
solution = np.zeros( (n,n), int )
for i in range(n):
    for j in range(n):
        solution[i,j] = int(x[i,j].value)
print(solution)
```

結果は以下のように表示される．
```
逸脱制約= {}
[[2 9 4]
 [7 5 3]
 [6 1 8]]
```
<!---->
### 多制約ナップサック

あなたは，ぬいぐるみ専門の泥棒だ．
ある晩，あなたは高級ぬいぐるみ店にこっそり忍び込んで，盗む物を選んでいる．
狙いはもちろん，マニアの間で高額で取り引きされているクマさん人形だ．
クマさん人形は，現在 $4$体販売されていて，
それらの値段と重さと容積は，以下のリストで与えられている．
```python
v=[16,19,23,28]                     #価値
a=[[2,3,4,5],[3000,3500,5100,7200]] #重さと容積
```
あなたは，転売価格の合計が最大になるようにクマさん人形を選んで逃げようと思っているが，
あなたが逃走用に愛用しているナップサックはとても古く，
$7$kgより重い荷物を入れると，底がぬけてしまうし，$10000 {cm}^3$（$10$$\ell$）を超えた荷物を入れると破けてしまう．

さて，どのクマさん人形をもって逃げれば良いだろうか？

```python
model=Model()

v=[16,19,23,28]
a=[[2,3,4,5],[3000,3500,5100,7200]]
b=[7,10000]
n=len(v)
m=len(b)
items=["item{0}".format(j) for j in range(n)]
varlist=model.addVariables(items,[0,1])
for i in range(m):
    con1=Linear("mkp_{0}".format(i),"inf",b[i])
    for j in range(n):
        con1.addTerms(a[i][j],varlist[j],1)
    model.addConstraint(con1)

con2=Linear("obj",1,sum(v),">=")
for j in range(n):
    con2.addTerms(v[j],varlist[j],1)
model.addConstraint(con2)

model.Params.TimeLimit=1
sol,violated=model.optimize()

print (model)

if model.Status==0:
    print ("solution")
    for x in sol:
        print (x,sol[x])
    print ("violated constraint(s)")
    for v in violated:
        print (v,violated[v])
```

結果は以下のように表示される．

```python
Model:
number of variables = 4
number of constraints= 3
variable item0:['0', '1'] = 0
variable item1:['0', '1'] = 1
variable item2:['0', '1'] = 1
variable item3:['0', '1'] = 0
mkp_0: weight= inf type=linear 2(item0,1) 3(item1,1) 4(item2,1) 5(item3,1) <=7 :LHS =7
mkp_1: weight= inf type=linear 3000(item0,1) 3500(item1,1) 5100(item2,1) 7200(item3,1) <=10000 :LHS =8600
obj: weight= 1 type=linear 16(item0,1) 19(item1,1) 23(item2,1) 28(item3,1) >=86 :LHS =42
solution
item0 0
item1 1
item2 1
item3 0
violated constraint(s)
obj 44
```
<!---->
### 最大安定集合


あなたは $6$人のお友達から何人か選んで一緒にピクニックに行こうと思っている．
しかし，グラフ上で隣接している（線で結ばれている）人同士はとても仲が悪く，彼らが一緒にピクニックに
行くとせっかくの楽しいピクニックが台無しになってしまう．
なるべくたくさんの仲間でピクニックに行くには誰を誘えばいいんだろう？

ただし，グラフの隣接点の情報は以下のリストで与えられているものとする．
```
adj=[[2],[3],[0,3,4,5],[1,2,5],[2],[2,3]]
```

```python
m=Model()

nodes=["n{0}".format(i) for i in range(6)]
adj=[[2],[3],[0,3,4,5],[1,2,5],[2],[2,3]]
n=len(nodes)

varlist=m.addVariables(nodes,[0,1])

for i in range(n):
    for j in adj[i]:
        if i<j:
            con1=Linear("constraint{0}_{1}".format(i,j),"inf",1)
            con1.addTerms(1,varlist[i],1)
            con1.addTerms(1,varlist[j],1)
            m.addConstraint(con1)

obj=Linear("obj",1,n,">=")
for i in range(n):
    obj.addTerms(1,varlist[i],1)
m.addConstraint(obj)

m.Params.TimeLimit=1
sol,violated=m.optimize()

print (m)

if m.Status==0:
    print ("solution")
    for x in sol:
        print (x,sol[x])
    print ("violated constraint(s)")
    for v in violated:
        print (v,violated[v])
```

結果は以下のように表示される．

```python
Model:
number of variables = 6
number of constraints= 7
variable n0:['0', '1'] = 1
variable n1:['0', '1'] = 1
variable n2:['0', '1'] = 0
variable n3:['0', '1'] = 0
variable n4:['0', '1'] = 1
variable n5:['0', '1'] = 1
constraint0_2: weight= inf type=linear 1(n0,1) 1(n2,1) <=1 :LHS =1
constraint1_3: weight= inf type=linear 1(n1,1) 1(n3,1) <=1 :LHS =1
constraint2_3: weight= inf type=linear 1(n2,1) 1(n3,1) <=1 :LHS =0
constraint2_4: weight= inf type=linear 1(n2,1) 1(n4,1) <=1 :LHS =1
constraint2_5: weight= inf type=linear 1(n2,1) 1(n5,1) <=1 :LHS =1
constraint3_5: weight= inf type=linear 1(n3,1) 1(n5,1) <=1 :LHS =1
obj: weight= 1 type=linear 1(n0,1) 1(n1,1) 1(n2,1) 1(n3,1) 1(n4,1) 1(n5,1) >=6 :LHS =4
solution
n0 1
n1 1
n2 0
n3 0
n4 1
n5 1
violated constraint(s)
obj 2
```
<!---->
### グラフ彩色

今度は，同じお友達のクラス分けで悩んでいる．
お友達同士で仲が悪い組は，グラフ上で隣接している．
仲が悪いお友達を同じクラスに入れると喧嘩を始めてしまう．
なるべく少ないクラスに分けるには，どのようにすればいいんだろう？

ただし，グラフの隣接点の情報は以下のリストで与えられているものとする．
```
adj=[[2],[3],[0,3,4,5],[1,2,5],[2],[2,3]]
```

```python
m=Model()

K=3
nodes=["n{0}".format(i) for i in range(6)]
adj=[[2],[3],[0,3,4,5],[1,2,5],[2],[2,3]]
n=len(nodes)

varlist=m.addVariables(nodes,range(K))

for i in range(n):
    for j in adj[i]:
        if i<j:
            con1=Alldiff("alldiff_{0}_{1}".format(i,j),[varlist[i],varlist[j]],"inf")
            m.addConstraint(con1)

m.Params.TimeLimit=1
sol,violated=m.optimize()

print (m)
if m.Status==0:
    print ("solution")
    for x in sol:
        print (x,sol[x])
    print ("violated constraint(s)")
    for v in violated:
        print (v,violated[v])
```

結果は以下のように表示される．

```python
Model:
number of variables = 6
number of constraints= 6
variable n0:['0', '1', '2'] = 0
variable n1:['0', '1', '2'] = 0
variable n2:['0', '1', '2'] = 1
variable n3:['0', '1', '2'] = 2
variable n4:['0', '1', '2'] = 2
variable n5:['0', '1', '2'] = 0
alldiff_0_2: weight= inf type=alldiff  n2 n0 ;  :LHS =0
alldiff_1_3: weight= inf type=alldiff  n3 n1 ;  :LHS =0
alldiff_2_3: weight= inf type=alldiff  n2 n3 ;  :LHS =0
alldiff_2_4: weight= inf type=alldiff  n2 n4 ;  :LHS =0
alldiff_2_5: weight= inf type=alldiff  n2 n5 ;  :LHS =0
alldiff_3_5: weight= inf type=alldiff  n3 n5 ;  :LHS =0
solution
n0 0
n1 0
n2 1
n3 2
n4 2
n5 0
violated constraint(s)
```
<!---->
### グラフ分割

今度は，同じ$6$人のお友達を2つのチームに分けてミニサッカーをしようとしている．
もちろん，公平を期すために，同じ人数になるように3人ずつに分ける．
ただし，仲が悪いお友達が同じチームになることは極力避けたいと考えている．
さて，どのようにチーム分けをしたら良いだろうか？

ただし，中の悪い同士を表すグラフの隣接点の情報は以下のリストで与えられているものとする．
```python
adj=[[1,4],[0,2,4],[1],[4,5],[0,1,3,5],[3,4]]
```

```python
nodes=[f"n{i}" for i in range(6)]
adj=[[1,4],[0,2,4],[1],[4,5],[0,1,3,5],[3,4]]
n=len(nodes)

m = Model()

varlist=m.addVariables(nodes,[0,1])

con1=Linear("constraint","inf",n//2,"=")
for i in range(len(nodes)):
    con1.addTerms(1,varlist[i],1)
m.addConstraint(con1)

##con2={}
##for i in range(n):
##    for j in adj[i]:
##        con2[i,j]= Quadratic( "obj_%s_%s"%(i,j) )
##        con2[i,j].addTerms(1,varlist[i],1,varlist[j],0)
##        con2[i,j].addTerms(1,varlist[i],0,varlist[j],1)
##        m.addConstraint(con2[i,j])

con2=Quadratic( "obj")
for i in range(n):
    for j in adj[i]:
        con2.addTerms(1,varlist[i],1,varlist[j],0)
        con2.addTerms(1,varlist[i],0,varlist[j],1)
m.addConstraint(con2)

m.Params.TimeLimit=1
sol,violated=m.optimize()

print(m)

if m.Status==0:
    print ("solution")
    for x in sol:
        print (x,sol[x])
    print ("violated constraint(s)")
    for v in violated:
        print (v,violated[v])
```

結果は以下のように表示される．

```python
Model:
number of variables = 6
number of constraints= 2
variable n0:['0', '1'] = None
variable n1:['0', '1'] = None
variable n2:['0', '1'] = None
variable n3:['0', '1'] = None
variable n4:['0', '1'] = None
variable n5:['0', '1'] = None
constraint: weight= inf type=linear 1(n0,1) 1(n1,1) 1(n2,1) 1(n3,1) 1(n4,1) 1(n5,1) =3 :LHS =0
obj: weight=1 type=quadratic 1(n0,1)(n1,0) 1(n0,0)(n1,1) 1(n0,1)(n4,0) 1(n0,0)(n4,1) 1(n1,1)(n0,0) 1(n1,0)(n0,1) 1(n1,1)(n2,0) 1(n1,0)(n2,1) 1(n1,1)(n4,0) 1(n1,0)(n4,1) 1(n2,1)(n1,0) 1(n2,0)(n1,1) 1(n3,1)(n4,0) 1(n3,0)(n4,1) 1(n3,1)(n5,0) 1(n3,0)(n5,1) 1(n4,1)(n0,0) 1(n4,0)(n0,1) 1(n4,1)(n1,0) 1(n4,0)(n1,1) 1(n4,1)(n3,0) 1(n4,0)(n3,1) 1(n4,1)(n5,0) 1(n4,0)(n5,1) 1(n5,1)(n3,0) 1(n5,0)(n3,1) 1(n5,1)(n4,0) 1(n5,0)(n4,1) <=0 :LHS =0

solution
n0 0
n1 0
n2 0
n3 1
n4 1
n5 1
violated constraint(s)
obj 4
```
<!---->
### 巡回セールスマン

あなたは休暇を利用してヨーロッパめぐりをしようと考えている．
現在スイスのチューリッヒに宿を構えているあなたの目的は，
スペインのマドリッドで闘牛を見ること，
イギリスのロンドンでビックベンを見物すること，
イタリアのローマでコロシアムを見ること，
ドイツのベルリンで本場のビールを飲むことである．

あなたはレンタルヘリコプターを借りてまわることにしたが，
移動距離に比例した高額なレンタル料を支払わなければならない．
したがって，
あなたはチューリッヒ (T) を出発した後，
なるべく短い距離で他の $4$つの都市 マドリッド(M)，ロンドン(L)，ローマ(R)，ベルリン(B) を経由し，
再びチューリッヒに帰って来ようと考えた．
都市の間の移動距離を測ってみたところ，以下のようになっていることがわかった．
```python
cities=["T","L","M","R","B"]
d=[[0,476,774,434,408],
   [476,0,784,894,569],
   [774,784,0,852,1154],
   [434,894,852,0,569],
   [408,569,1154,569,0]]
```
さて，どのような順序で旅行すれば，移動距離が最小になるだろうか?

```python
m=Model()

cities=["T","L","M","R","B"]
d=[[0,476,774,434,408],[476,0,784,894,569],[774,784,0,852,1154],[434,894,852,0,569],[408,569,1154,569,0]]
n=len(cities)

varlist=m.addVariables(cities,range(n))

con1=Alldiff("AD",varlist,"inf")
m.addConstraint(con1)

obj=Quadratic("obj")
for i in range(n):
    for j in range(n):
        if i!=j:
            for k in range(n):
                if k ==n-1:
                    ell=0
                else:
                    ell=k+1
                obj.addTerms(d[i][j],varlist[i],k,varlist[j],ell)
m.addConstraint(obj)

m.Params.TimeLimit=1
sol,violated=m.optimize()

print (m)

if m.Status==0:
    print ("solution")
    for x in sol:
        print (x,sol[x])
    print ("violated constraint(s)")
    for v in violated:
        print (v,violated[v])
```

結果は以下のように表示される．

```python
Model:
number of variables = 5
number of constraints= 2
variable T:['0', '1', '2', '3', '4'] = 1
variable L:['0', '1', '2', '3', '4'] = 3
variable M:['0', '1', '2', '3', '4'] = 4
variable R:['0', '1', '2', '3', '4'] = 0
variable B:['0', '1', '2', '3', '4'] = 2
AD: weight= inf type=alldiff  M L T B R ;  :LHS =0
obj: weight=1 type=quadratic 476(T,0)(L,1) 476(T,1)(L,2) 476(T,2)(L,3) 476(T,3)(L,4) 476(T,4)(L,0) 774(T,0)(M,1) 774(T,1)(M,2) 774(T,2)(M,3) 774(T,3)(M,4) 774(T,4)(M,0) 434(T,0)(R,1) 434(T,1)(R,2) 434(T,2)(R,3) 434(T,3)(R,4) 434(T,4)(R,0) 408(T,0)(B,1) 408(T,1)(B,2) 408(T,2)(B,3) 408(T,3)(B,4) 408(T,4)(B,0) 476(L,0)(T,1) 476(L,1)(T,2) 476(L,2)(T,3) 476(L,3)(T,4) 476(L,4)(T,0) 784(L,0)(M,1) 784(L,1)(M,2) 784(L,2)(M,3) 784(L,3)(M,4) 784(L,4)(M,0) 894(L,0)(R,1) 894(L,1)(R,2) 894(L,2)(R,3) 894(L,3)(R,4) 894(L,4)(R,0) 569(L,0)(B,1) 569(L,1)(B,2) 569(L,2)(B,3) 569(L,3)(B,4) 569(L,4)(B,0) 774(M,0)(T,1) 774(M,1)(T,2) 774(M,2)(T,3) 774(M,3)(T,4) 774(M,4)(T,0) 784(M,0)(L,1) 784(M,1)(L,2) 784(M,2)(L,3) 784(M,3)(L,4) 784(M,4)(L,0) 852(M,0)(R,1) 852(M,1)(R,2) 852(M,2)(R,3) 852(M,3)(R,4) 852(M,4)(R,0) 1154(M,0)(B,1) 1154(M,1)(B,2) 1154(M,2)(B,3) 1154(M,3)(B,4) 1154(M,4)(B,0) 434(R,0)(T,1) 434(R,1)(T,2) 434(R,2)(T,3) 434(R,3)(T,4) 434(R,4)(T,0) 894(R,0)(L,1) 894(R,1)(L,2) 894(R,2)(L,3) 894(R,3)(L,4) 894(R,4)(L,0) 852(R,0)(M,1) 852(R,1)(M,2) 852(R,2)(M,3) 852(R,3)(M,4) 852(R,4)(M,0) 569(R,0)(B,1) 569(R,1)(B,2) 569(R,2)(B,3) 569(R,3)(B,4) 569(R,4)(B,0) 408(B,0)(T,1) 408(B,1)(T,2) 408(B,2)(T,3) 408(B,3)(T,4) 408(B,4)(T,0) 569(B,0)(L,1) 569(B,1)(L,2) 569(B,2)(L,3) 569(B,3)(L,4) 569(B,4)(L,0) 1154(B,0)(M,1) 1154(B,1)(M,2) 1154(B,2)(M,3) 1154(B,3)(M,4) 1154(B,4)(M,0) 569(B,0)(R,1) 569(B,1)(R,2) 569(B,2)(R,3) 569(B,3)(R,4) 569(B,4)(R,0) <=0 :LHS =3047
solution
T 1
L 3
M 4
R 0
B 2
violated constraint(s)
obj 3047

```
<!---->
### ビンパッキング

あなたは，大企業の箱詰め担当部長だ．あなたの仕事は，色々な大きさのものを，決められた大きさの箱に「上手に」詰めることである．
この際，使う箱の数をなるべく少なくすることが，あなたの目標だ．
（なぜって，あなたの会社が利用している宅配業者では，運賃は箱の数に比例して決められるから．）
1つの箱に詰められる荷物の上限は $7$kgと決まっており，荷物の重さはのリストは
[6,5,4,3,1,2] である．
しかも，あなたの会社で扱っている荷物は，どれも重たいものばかりなので，容積は気にする必要はない
（すなわち箱の容量は十分と仮定する）．
さて，どのように詰めて運んだら良いだろうか？


```python
bpp=Model()

Items=[6,5,4,3,1,2]
B=7
num_bins=3
n=len(Items)

x={}
for i in range(n):
     x[i] = bpp.addVariable("x_{0}".format(i),range(num_bins))
Bin={}
for j in range(num_bins):
     Bin[j]=Linear("Bin_{0}".format(j),weight=1,rhs=B,direction="<=")
     for i in range(n):
          Bin[j].addTerms(Items[i],x[i],j)
     bpp.addConstraint(Bin[j])

sol,violated=bpp.optimize()

if bpp.Status==0:
    print ("solution=")
    for i in sol:
        print (i,sol[i])

    print ("violated constraints=",violated)
```

結果は以下のように表示される．

```python
Model:
number of variables = 6
number of constraints= 3
variable x_0:['0', '1', '2'] = 2
variable x_1:['0', '1', '2'] = 0
variable x_2:['0', '1', '2'] = 1
variable x_3:['0', '1', '2'] = 1
variable x_4:['0', '1', '2'] = 2
variable x_5:['0', '1', '2'] = 0
Bin_0: weight= 1 type=linear 6(x_0,0) 5(x_1,0) 4(x_2,0) 3(x_3,0) 1(x_4,0) 2(x_5,0) <=7 :LHS =7
Bin_1: weight= 1 type=linear 6(x_0,1) 5(x_1,1) 4(x_2,1) 3(x_3,1) 1(x_4,1) 2(x_5,1) <=7 :LHS =7
Bin_2: weight= 1 type=linear 6(x_0,2) 5(x_1,2) 4(x_2,2) 3(x_3,2) 1(x_4,2) 2(x_5,2) <=7 :LHS =7
solution=
x_0 2
x_1 0
x_2 1
x_3 1
x_4 2
x_5 0
violated constraints= {}
```
<!---->
### 最適化版の$8$-クイーン

$8 \times 8$ のチェス盤に $8$個のクイーンを置くことを考える．
チェスのクイーンとは，将棋の飛車と角の両方の動きができる最強の駒である．
$i$行 $j$列に置いたときの費用を $i \times j$ と定義したとき，
クイーンがお互いに取り合わないように置く配置の中で，費用の合計が最小になるような配置を求めよ．


```python
m=Model()

n=8
varlist=[]
for i in range(n):
    varlist.append("x{0}".format(i))

var=m.addVariables(varlist,range(n))

con1=Alldiff("AD",var,"inf")
m.addConstraint(con1)

for k in range(2,2*n-1):
    con2=Linear("rightdown_{0}".format(k),"inf",1,"<=")
    for i in range(n):
        j=k-n+i
        if j>=0 and j<=n-1:
            con2.addTerms(1,var[i],j)
    m.addConstraint(con2)

for k in range(2,2*n-1):
    con3=Linear("leftdown_{0}".format(k),"inf",1,"<=")
    for i in range(n):
        j=k-i-1
        if j>=0 and j<=n-1:
            con3.addTerms(1,var[i],j)
    m.addConstraint(con3)

obj=Linear("obj",1,0,"<=")
for i in range(n):
    for j in range(n):
        obj.addTerms((i+1)*(j+1),var[i],j)
m.addConstraint(obj)

m.Params.TimeLimit=1
sol,violated=m.optimize()

if m.Status==0:
    print ("solution")
    for x in sol:
        print (x,sol[x])
    print ("violated constraint(s)")
    for v in violated:
        print (v,violated[v])
```

結果は以下のように表示される．

```python
solution
x0 6
x1 3
x2 1
x3 7
x4 5
x5 0
x6 2
x7 4
violated constraint(s)
obj 150
```
<!---->
### 2次割当

いま，3人のお友達が3箇所の家に住もうとしている．
3人は毎週何回か重要な打ち合わせをする必要があり，打ち合わせの頻度は，リストのリスト．
```python
f = [[0,5,1],[5,0,2],[1,2,0]]
```
として与えられている．

また，家の間の移動距離もリストのリスト
```python
d = [[0,2,3],[2,0,1],[3,1,0]]
```
として与えられているものとする．

3人は打ち合わせのときに移動する距離を最小に
するような場所に住むことを希望している．さて，誰をどの家に割り当てたらよいのだろうか？

```python
m=Model()

n=3
d=[[0,2,3],[2,0,1],[3,1,0]]
f=[[0,5,1],[5,0,2],[1,2,0]]

nodes=["n{0}".format(i) for i in range(n)]

varlist=m.addVariables(nodes,range(n))

con1=Alldiff("AD",varlist,"inf")
m.addConstraint(con1)

obj=Quadratic("obj")
for i in range(n-1):
    for j in range(i+1,n):
        for k in range(n):
            for ell in range(n):
                if k !=ell:
                    obj.addTerms(f[i][j]*d[k][ell],varlist[i],k,varlist[j],ell)
m.addConstraint(obj)

m.Params.TimeLimit=1
sol,violated=m.optimize()

if m.Status==0:
    print ("solution")
    for x in sol:
        print (x,sol[x])
    print ("violated constraint(s)")
    for v in violated:
        print (v,violated[v])
```

結果は以下のように表示される．

```python
solution
n0 2
n1 1
n2 0
violated constraint(s)
obj 12
```
<!---->
### $k$-メディアン

顧客から最も近い施設への距離の「合計」を最小にするように
グラフ内の点上または空間内の任意の点から施設を選択する問題である．
メディアン問題においては，選択される施設の数があらかじめ決められていることが多く，
その場合には選択する施設数 $k$ を頭につけて **$k$-メディアン問題**($k$-median problem)とよばれる．

顧客数を $n$ とし，顧客の集合を $I$，施設の配置可能な点の集合を $J$ とする．
簡単のため $I=J$ とし， 2地点間の移動距離を以下のデータ生成関数によって生成したものとしたとき， 地点数 $|I|=|J|=200$， $k=20$ の $k$-メディアン問題の解を求めよ．

注意：この問題はトライアル版では解けない．

```python
import random
import math
import networkx as nx

def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

random.seed(67)
def make_data(n,m,same=True):
    if same == True:
        I = range(n)
        J = range(m)
        x = [random.random() for i in range(max(m,n))]
        # positions of the points in the plane
        y = [random.random() for i in range(max(m,n))]
    else:
        I = range(n)
        J = range(n,n+m)
        x = [random.random() for i in range(n+m)]
        # positions of the points in the plane
        y = [random.random() for i in range(n+m)]
    c = {}
    for i in I:
        for j in J:
            c[i,j] = distance(x[i],y[i],x[j],y[j])

    return I,J,c,x,y

n = 200
m = n
I, J, c, x_pos, y_pos = make_data(n,m,same=True)
k = 20

I_list=[]
for i in I:
    I_list.append(f"c{i}")

J_list=[]
for j in J:
    J_list.append(f"f{j}")

cost={}
for i in I:
    for j in J:
        cost[f"c{i}",f"f{j}"]= int(100*c[i,j])

m=Model()
x={}
y={}
for i in I_list:
    x[i]=m.addVariable(name=i,domain=J_list)

for j in J_list:
    y[j]=m.addVariable(name=j,domain=[0,1])

con1=Linear('con1',weight="inf",rhs=k,direction='=')
for j in J_list:
    con1.addTerms(1,y[j],1)
m.addConstraint(con1)

con2={}
for j in J_list:
    for i in I_list:
        con2[i,j]=Linear(f'con2_{i}_{j}',weight="inf",rhs=0,direction='<=')
        con2[i,j].addTerms(1,x[i],j)
        con2[i,j].addTerms(-1,y[j],1)
        m.addConstraint(con2[i,j])

obj=Linear('obj',weight=1,rhs=0,direction='<=')
for i,j in cost:
    obj.addTerms(cost[i,j],x[i],j)

m.addConstraint(obj)

m.Params.TimeLimit=100
sol,violated=m.optimize()

edges=[]
facilities=[]

print('solution')
for x in sol:
    #print (x,sol[x])
    if x[0]=="c":
        edges.append((int(x[1:]),int(sol[x][1:])))

    elif x[0]=="f" and sol[x]=="1":
        facilities.append(int(x[1:]))

print ('violated constraint(s)')
for v in violated:
    print (v,violated[v])

position = {}
for i in range(len(x_pos)):
    position[i] = (x_pos[i],y_pos[i])

G = nx.Graph()
facilities = set(facilities)
unused = set(j for j in J if j not in facilities)
client = set(i for i in I if i not in facilities and i not in unused)
G.add_nodes_from(facilities)
G.add_nodes_from(client)
G.add_nodes_from(unused)
for (i,j) in edges:
    G.add_edge(i,j)

nx.draw(G,position,with_labels=False,node_color="b",
        nodelist=facilities)
nx.draw(G,position,with_labels=False,node_color="c",
        nodelist=unused,node_size=50)
nx.draw(G,position,with_labels=False,node_color="g",
        nodelist=client,node_size=50)
```

結果は以下のように表示される．

```python
solution
violated constraint(s)
obj 1465
```

<img src="https://github.com/scmopt/scmopt_data/blob/main/fig/scop-kmedian.png?raw=true" width=600 height=200>
<!---->
### 数独パズル

与えられる盤面は 3 × 3 のブロックに区切られた 9 × 9 の格子である.その幾つかのマスには，あらかじめ 1 から 9 までの数字が入っている.
与えられた盤面の空いているマスに次の規則を満たすように 1 から 9 までの数字を入れる.

- どの横行，縦列，および太線で囲まれた 3 × 3 のブロックにも同じ数字が入ってはいけない.

これは，そのどれらにもちょうど 1 つずつ 1 から 9 までの数字が入ることと同じ意味である.

注意：この問題はトライアル版では解けない．

```python
grid1 = """
.14.5...3
6....942.
8..1...9.
..5.9..4.
4..7.8..2
.7..2.6..
.9...1..5
.283....4
5...6.71.
"""
def values_from_grid(grid):
    values = []
    digits = "123456789"
    chars = [c for c in grid if c in digits or c in '0.']
    grid_int = map(lambda x: int(x) if x != "." else 0, chars)

    count = 0
    row = []
    for i in grid_int:
        row.append(i)
        count += 1
        if count % 9 == 0: #行毎に分割
            values.append(row)
            row = []
    return values
value = values_from_grid(grid1)
print(value)

model = Model()
x = {}
n = 9
dom = [i+1 for i in range(n)]
for i in range(n):
    for j in range(n):
        x[i,j] = model.addVariable(name=f"x[{i},{j}]", domain=dom)
row_constr = {}
for i in range(n):
    row_constr[i] = Alldiff(f'row{i}',[ x[i,j] for j in range(n) ], weight='inf')
    model.addConstraint( row_constr[i] )
col_constr = {}
for j in range(n):
    col_constr[j] = Alldiff(f'col{j}',[ x[i,j] for i in range(n) ], weight='inf')
    model.addConstraint( col_constr[j] )
block_constr = {}
nb = n//3 #ブロック数
for k in range(nb):
    for l in range(nb):
        block_constr[k,l] = Alldiff(f'block{k},{l}',[ x[k*nb+i,l*nb+j] for i in range(nb) for j in range(nb) ], weight='inf')
        model.addConstraint( block_constr[k,l] )
binding_constr = {}
for i in range(n):
    for j in range(n):
        if value[i][j] != 0:
            binding_constr[i,j] = Linear(f'linear_constraint{i},{j}',weight=1,rhs=1,direction='=')
            binding_constr[i,j].addTerms(1,x[i,j], value[i][j])
            model.addConstraint(binding_constr[i,j])
model.Params.TimeLimit=1
model.Params.RandomSeed=1
sol,violated = model.optimize()
print("逸脱制約=", violated)
import numpy as np
solution = np.zeros( (n,n), int )
for i in range(n):
    for j in range(n):
        solution[i,j] = int(x[i,j].value)
print(solution)
```

結果は以下のように表示される．

```python
[[0, 1, 4, 0, 5, 0, 0, 0, 3],
 [6, 0, 0, 0, 0, 9, 4, 2, 0],
 [8, 0, 0, 1, 0, 0, 0, 9, 0],
 [0, 0, 5, 0, 9, 0, 0, 4, 0],
 [4, 0, 0, 7, 0, 8, 0, 0, 2],
 [0, 7, 0, 0, 2, 0, 6, 0, 0],
 [0, 9, 0, 0, 0, 1, 0, 0, 5],
 [0, 2, 8, 3, 0, 0, 0, 0, 4],
 [5, 0, 0, 0, 6, 0, 7, 1, 0]]
逸脱制約= {}
[[9 1 4 2 5 6 8 7 3]
 [6 5 7 8 3 9 4 2 1]
 [8 3 2 1 4 7 5 9 6]
 [2 8 5 6 9 3 1 4 7]
 [4 6 9 7 1 8 3 5 2]
 [3 7 1 5 2 4 6 8 9]
 [7 9 6 4 8 1 2 3 5]
 [1 2 8 3 7 5 9 6 4]
 [5 4 3 9 6 2 7 1 8]]
```
<!---->
### シフトスケジューリング

あなたは，24時間営業のハンバーガーショップのオーナーであり，スタッフの1週間のシフトを組むことに頭を悩ませている．
スタッフの時給は同じであると仮定したとき，以下の制約を満たすシフトを求めよ．

-  1シフトは 8時間で，朝，昼，晩の3シフトの交代制とする．
-  3人のスタッフは，1日に高々1つのシフトしか行うことができない．
-  繰り返し行われる1週間のスケジュールの中で，スタッフは最低3日間は勤務しなければならない．
-  スタッフの夜勤は最大で4回とする．
-  各スタッフは1日以上休みを取る必要がある．
-  各シフトに割り当てられるスタッフの数は，ちょうど 1人でなければならない．
-  異なるシフトを翌日に行ってはいけない．（すなわち異なるシフトに移るときには，必ず休日を入れる必要がある．）
-  シフト2, 3は，少なくとも2日間は連続で行わなければならない．

注意：この問題はトライアル版では解けない．

```python
periods=[1,2,3,4,5]
shifts=[0,1,2]
staffs=["A","B","C"]

m = Model()
var={} #list of variables
for i in staffs:
     for t in periods:
          var[i,t]=m.addVariable(name=i+str(t),domain=shifts)

LB={} #各スタッフは最低3日以上出勤する必要がある．
for i in staffs:
     LB[i]=Linear("LB_{0}".format(i),rhs=3,direction=">=") #weight is set to default (1)
     for t in periods:
          for s in range(1,len(shifts)):
               LB[i].addTerms(1,var[i,t],shifts[s])
     m.addConstraint(LB[i])

UB={} #各スタッフは1日以上休みを取る必要がある．
for i in staffs:
     UB[i]=Linear("UB_{0}".format(i),rhs=6,direction="<=") #weight is set to default (1)
     for t in periods:
          for s in range(1,len(shifts)):
               UB[i].addTerms(1,var[i,t],shifts[s])
     m.addConstraint(UB[i])

UB_night={} #各スタッフの夜勤回数は最大4回まで可能．
for i in staffs:
     UB_night[i]=Linear("UB_night_{0}".format(i),rhs=4,direction="<=") #weight is set to default (1)
     for t in periods:
          UB_night[i].addTerms(1,var[i,t],shifts[-1])
     m.addConstraint(UB_night[i])

UB_shift={} #各シフトには1人のスタッフを割り当てる必要がある．
for t in periods:
     for s in range(1,len(shifts)):
          UB_shift[t,s]=Linear("UBshift_{0}_{1}".format(t,s),rhs=1,direction="=") #weight is set to default (1)
          for i in staffs:
               UB_shift[t,s].addTerms(1,var[i,t],shifts[s])
          m.addConstraint(UB_shift[t,s])

#異なるシフトに移る場合は休みを入れる必要がある．（異なるシフトが2日間連続で行うのを禁止する制約．）
Forbid={}
for i in staffs:
     for t in periods:
          for s in range(1,len(shifts)):
            Forbid[(i,t,s)]=Linear("Forbid_{0}_{1}_{2}".format(i,t,s),rhs=1)
            Forbid[(i,t,s)].addTerms(1,var[i,t],shifts[s])
            for k in range(1,len(shifts)):
                if k!=s:
                    if t==periods[-1]:
                         Forbid[(i,t,s)].addTerms(1,var[i,1],shifts[k])
                    else:
                         Forbid[(i,t,s)].addTerms(1,var[i,t+1],shifts[k])
            m.addConstraint(Forbid[(i,t,s)])

#シフト「昼」，「夜」は，最低2日間は連続で行う．
Cons={}
for i in staffs:
     for t in periods:
        for s in range(2,len(shifts)):
             Cons[(i,t)]=Linear("Cons_{0}_{1}".format(i,t),direction=">=")
             Cons[(i,t)].addTerms(-1,var[i,t],shifts[s])
             if t==1:
                  Cons[(i,t)].addTerms(1,var[i,periods[-1]],shifts[s])
             else:
                  Cons[(i,t)].addTerms(1,var[i,t-1],shifts[s])
             if t==periods[-1]:
                  Cons[(i,t)].addTerms(1,var[i,1],shifts[s])
             else:
                  Cons[(i,t)].addTerms(1,var[i,t+1],shifts[s])
             m.addConstraint(Cons[(i,t)])


m.Params.TimeLimit=1
sol,violated=m.optimize()

#print (m)
if m.Status==0:
    print ("solution")
    for x in sol:
        print (x,sol[x])
    print ("violated constraint(s)")
    for v in violated:
        print (v,violated[v])
```

結果は以下のように表示される．

```python
solution
A1 2
A2 2
A3 0
A4 0
A5 2
B1 0
B2 1
B3 1
B4 1
B5 1
C1 1
C2 0
C3 2
C4 2
C5 0
violated constraint(s)
```
<!---->
### 車の投入順決定

コンベア上に一直線に並んだ車の生産ラインを考える．
このラインは，幾つかの作業場から構成され，それぞれの作業場では異なる作業が行われる．
いま，4種類の車を同じ生産ラインで製造しており，それぞれをモデル $A,B,C,D$ とする．
本日の製造目標は，それぞれ $30,30,20,40$台である．

最初の作業場では，サンルーフの取り付けを行っており，これはモデル $B,C$ だけに必要な作業である．
次の作業場では，カーナビの取り付けが行われており，これはモデル $A,C$ だけに必要な作業である．
それぞれの作業場は長さをもち，
サンルーフ取り付けは車 $5$台分，カーナビ取り付けは車 $3$台分の長さをもつ．
また，作業場には作業員が割り当てられており，サンルーフ取り付けは $3$人，カーナビ取り付けは $2$人の
作業員が配置されており，作業場の長さを超えない範囲で別々に作業を行う．

作業場の範囲で作業が可能な車の投入順序を求めよ．

ヒント： 投入順序をうまく決めないと，作業場の範囲内で作業を完了することができない．
たとえば，$C,A,A,B,C$ の順で投入すると，
サンルーフ取り付けでは，3人の作業員がそれぞれモデル $C,B,C$ に対する作業を行うので
間に合うが，カーナビ取り付けでは， 2人の作業員では $C,A,A$ の3台の車の作業を終えることができない．

これは，作業場の容量制約とよばれ，サンルーフ取り付けの作業場では，
すべての連続する $5$台の車の中に，モデル $B,C$ が高々 $3$つ，
カーナビ取り付けの作業場では，
すべての連続する $3$台の車の中に，モデル $A,C$ が高々 $2$つ入っているという制約を課すことに相当する

```python
m=Model()
Type=["A","B","C","D","E","F"] #car types
Number={"A":1,"B":1,"C":2,"D":2,"E":2,"F":2}   #number of cars needed
n=sum(Number[i] for i in Number) #planning horizon
#1st line produces car type B and C that has a workplace with length 5 and 3 workers
#2nd line produces car type A anc C that has a workplace with length 3 and 2 workers
Option=[["A","E","F"],
    ["C","D","F"],
    ["A","E"],
    ["A","B","D"],
    ["C"]]
Length=[2,3,3,5,5]
Capacity=[1,2,1,2,1]

X={}
for i in range(n):
    X[i]=m.addVariable("seq[{0}]".format(i),Type)

#production volume constraints
for i in Type:
    L1=Linear("req[{0}]".format(i),direction="=",rhs=Number[i])
    for j in range(n):
        L1.addTerms(1,X[j],i)
    m.addConstraint(L1)

for i in range(len(Length)):
    for k in range(n-Length[i]+1):
        L2=Linear("ub[{0}_{1}]".format(i,k),direction="<=",rhs=Capacity[i])
        for t in range(k,k+Length[i]):
            for j in range(len(Option[i])):
                L2.addTerms(1,X[t],Option[i][j])
        m.addConstraint(L2)

m.Params.TimeLimit=1
m.Params.OutputFlag=False
sol,violated=m.optimize()

if m.Status==0:
    print ("solution")
    for x in sol:
        print (x,sol[x])
    print ("violated constraint(s)")
    for v in violated:
        print (v,violated[v])
```

結果は以下のように表示される．

```python
solution
seq[0] A
seq[1] C
seq[2] F
seq[3] B
seq[4] F
seq[5] D
seq[6] E
seq[7] C
seq[8] D
seq[9] E
violated constraint(s)
```
<!---->
### 段取り費用付き生産計画

１つの生産ラインでa,bの２種類の製品を生産している．各期に生産できる製品は1つであり，生産はバッチで行われるため生産量は決まっている（辞書S）．
5期の需要量（辞書D）を満たすように，生産計画（どの期にどの製品を生産するか）を作りたいのだが，製品の切り替えには段取り費用（辞書F）がかかる．
ただし，生産しないことを表すダミーの製品０があるものと仮定し，直前の期では何も生産していなかったものと仮定する．
生産すると生産量だけ在庫が増え，毎期需要分だけ在庫が減少する．
初期在庫（辞書I0）を与えたとき，各期の在庫量が上限（辞書UB）以下，下限（辞書LB)以上でなければいけないとしたとき，段取り費用の合計を最小にする生産計画をたてよ．

```python
S={"0":0,"a":30,"b":50} #S[P,T]：単位生産量
UB={"0":0,"a":50,"b":50} #UB[p,t]：在庫量の上限
LB={"0":0,"a":10,"b":10}  #LB[p]：在庫量の下限
I0={"0":0,"a":10,"b":30} #I0[p]:初期在庫

#D[p,t]：需要量
D={('0',1):0,('0',2):0,('0',3):0,('0',4):0,('0',5):0,
   ('a',1):10,('a',2):10,('a',3):30,('a',4):10,('a',5):10,
   ('b',1):20,('b',2):10,('b',3):20,('b',4):10,('b',5):10}

#F[p,q]: 製品p,q間の段取り費用
F={('0',"a"):10,('0',"b"):10,
   ('a',"0"):10,('a',"b"):30,
   ('b',"0"):10,('b',"a"):10}
```

```python
prod=["0","a","b"]      #製品の種類
T=5                   #計画期間は5期

S={"0":0,"a":30,"b":50} #S[P,T]：単位生産量
UB={"0":0,"a":50,"b":50} #UB[p,t]：在庫量の上限
LB={"0":0,"a":10,"b":10}  #LB[p]：在庫量の下限
I0={"0":0,"a":10,"b":30} #I0[p]:初期在庫

#D[p,t]：需要量
D={('0',1):0,('0',2):0,('0',3):0,('0',4):0,('0',5):0,
   ('a',1):10,('a',2):10,('a',3):30,('a',4):10,('a',5):10,
   ('b',1):20,('b',2):10,('b',3):20,('b',4):10,('b',5):10}

#F[p,q]: 製品p,q間の段取り費用
F={('0',"a"):10,('0',"b"):10,
   ('a',"0"):10,('a',"b"):30,
   ('b',"0"):10,('b',"a"):10}

model=Model()

X={}          #X[p,t]：製品pを期tに生産するかどうかの0-1変数
for t in range(1,T+1):
    X[t]=model.addVariable("X{0}".format(t),prod)

#constraint
for p in prod:
    if p=="0":
        pass
    else:
        for t in range(1,T+1):
            D_temp=0
            for i in range(1,t+1):
                D_temp+=D[p,i]
            con1=Linear("LB{0}_{1}".format(p,t),"inf",LB[p]-I0[p]+D_temp,">=")
            for i in range(1,t+1):
                con1.addTerms(S[p],X[i],p)
            model.addConstraint(con1)

for p in prod:
    if p=="0":
        pass
    else:
        for t in range(1,T+1):
            D_temp=0
            for i in range(1,t+1):
                D_temp+=D[p,i]
            con2=Linear("UB{0}_{1}".format(p,t),"inf",UB[p]-I0[p]+D_temp,"<=")
            for i in range(1,t+1):
                con2.addTerms(S[p],X[i],p)
            model.addConstraint(con2)

for p in prod:
    if p=="0":
        pass
    else:
        for q in prod:
            if q=="0" or p==q:
                pass
            else:
                for t in range(2,T+1):
                    con3=Quadratic("obj{0}_{1}_{2}".format(p,q,t),1,0,"<=")
                    con3.addTerms(F[p,q],X[t-1],p,X[t],q)
                    model.addConstraint(con3)


model.Params.TimeLimit=1
sol,violated=model.optimize()

if model.Status==0:
    print ("solution")
    for x in sol:
        print (x,sol[x])
    print ("violated constraint(s)")
    for v in violated:
        print (v,violated[v])
```

結果は以下のように表示される．

```python
solution
X1 a
X2 b
X3 a
X4 a
X5 0
violated constraint(s)
obja_b_2 30
objb_a_3 10
```
<!---->
## 時間割作成問題

実践的な問題として時間割作成問題の例を示す．通常の時間割作成の他に，大学では試験の時間割などにもニーズがある．

次の集合が与えられている．

- 授業（クラス）集合: 大学への応用の場合には，授業には担当教師が紐付けられている．
- 教室集合
- 学生集合
- 期集合：通常は1週間(5日もしくは6日）の各時限を考える．

以下の条件を満たす時間割を求める．

- すべての授業をいずれか期へ割当
- すべての授業をいずれか教室へ割当
- 各期、1つの教室では1つ以下の授業
- 同じ教員の受持ち講義は異なる期へ
- 割当教室は受講学生数以上の容量をもつ
- 同じ学生が受ける可能性がある授業の集合（カリキュラム）は，異なる期へ割り当てなければならない

考慮すべき付加条件には，以下のものがある．

- 1日の最後の期に割り当てられると履修学生数分のペナルティ
- 1人の学生が履修する授業が3連続するとペナルティ
- 1人の学生の授業数が、1日に1つならばペナルティ（$0$ か $2$ 以上が望ましい）

これは制約最適化問題として以下のように定式化できる．

授業 $i$ の割り当て期を表す変数 $X_i$ （領域はすべての期の集合）と割り当て教室を表す変数 $Y_i$ （領域はすべての教室の集合）を用いる．

定式化では，これらの変数は値変数として表記する．
それぞれ，
授業 $i$ を期 $t$ に割り当てるとき $1$ の $0$-$1$ 変数 $x_{it}$， 授業 $i$ を教室 $k$ に割り当てるとき $1$ の $0$-$1$ 変数 $y_{ik}$ となる．

SCOPにおいては，以下の2つの制約は自動的に守られるので必要ない．

- すべての授業をいずれか期へ割当
- すべての授業をいずれか教室へ割当

他の制約は，以下のように記述できる．

- 各期 $t$ の各教室 $k$ への割り当て授業は1以下であることを表す制約：

-
$$
 \sum_{i} x_{it} y_{ik} \leq 1 \ \ \ \forall t,k
$$

- 同じ教員の受持ち講義は異なる期へ割り当て：

教員 $l$ の担当する授業の集合を $E_l$ とすると，この制約は 「$X_{i} (i \in E_l)$ はすべて異なる値をもつ」と書くことができる．
SCOPでは，このようなタイプの制約を相異制約とよび，そのまま記述できる．

- 割当教室は受講学生数以上の容量をもつ．

授業 $i$ ができない（容量を超過する）教室の集合を $K_i$ する．

$$
\sum_{i, k \in K_i} y_{ik}  \leq 0
$$

- 同じ学生が受ける可能性がある授業の集合（カリキュラム）は，異なる期へ割り当てなければならない．

カリキュラム $j$ に含まれる授業の集合を $C_j$ としたとき，以下の制約として記述できる．

$$
   \sum_{i \in C_j} x_{it} \leq 1 \ \ \ \forall j, t
$$

- 1日の最後の期に割り当てられると履修学生数分のペナルティ

1日の最後の期の集合を $L$， 授業 $i$ の履修学生数を $w_i$ とする．

$$
  \sum_{i, t \in L}  w_i x_{it} \leq 0
$$

- 1人の学生が履修する授業が3連続すると1ペナルティ

$T$ を1日のうちで最後の2時間でない期の集合とする．

$$
\sum_{i \in C_j} x_{it} + x_{i,t+1} +x_{i,t+2}  \leq 2 \ \ \ \forall j, t \in T
$$

- 1人の学生の授業数が、1日に1つならば1ペナルティ（0か2以上が望ましい）

各日 $d$ に含まれる期の集合を $T_d$ とし， 日 $d$ におけるカリキュラム $j$ に含まれる授業数が $0$ か $2$ 以上なのかを表す $0$-$1$ 変数 $z_{jd}$ を用いる．

$$
 \sum_{i \in C_j} x_{it} \leq |T_d| z_{jd}  \ \ \ \forall d, j
$$

$$
 \sum_{i \in C_j} x_{it} \geq 2 z_{jd}  \ \ \ \forall d, j
$$

上の定式化をSCOPで解くことによって，付加的な制約を「できるだけ」満たす時間割を作成することができる．

上の問題を通常の数理最適化ソルバーで記述しようとすると，多くの $0$-$1$変数を必要とし，非凸の2次制約を含むので現実的な時間で計算をすることが困難になる．
SCOPでは，変数の領域を用いたコンパクトな表現と，非凸の2次制約を直接最適化するため，効率的に近似解を算出することができる．

実際に，時間割作成の国際コンペティション (ITC2007) では， SCOPは3つの異なるトラックですべてメダル（3位, 2位, 3位）を獲得している．

```{.python.marimo}
import marimo as mo
```