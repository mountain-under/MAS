import random
import statistics
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt

class Worker:
    def __init__(self, production_capacity, firm):
        self.production_capacity = production_capacity  # 生産能力
        self.firm = firm  # 働く先の企業エージェント
        self.employed = False  # 雇用状態

        
# 家計エージェントのクラス
class HouseholdAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        self.total_population = random.randint(1, 6)  # 1から6人の世帯人数
        
        self.num_of_workers = random.randint(0, self.total_population)  # 労働者の人数
        if self.total_population - self.num_of_workers > 0:
            self.num_of_non_workers = random.randint(0, self.total_population - self.num_of_workers)  # 非労働者の人数
        else:
            self.num_of_non_workers = 0
        self.num_of_retirees = self.total_population - self.num_of_workers - self.num_of_non_workers  # 残りは年金受給者
        
        # 各労働者エージェントは生産能力と働く企業エージェントを属性として持つ
        self.workers = [Worker(random.randint(1, 5), None) for _ in range(self.num_of_workers)]
        self.income = 0 #収入
        self.disposable_income = 0 #可処分所得
        self.consumption = 0 #消費
        self.savings = 0 #貯蓄額

    #年金受給者が働く，労働者が年金受給者になるなどの操作は非常に煩雑なため，割愛．
        

    # 高賃金の企業を探す：賃金が現在の企業よりも高い企業を探し、その中からランダムに1つ選ぶ
    def find_higher_paying_job(self, worker):
        higher_paying_jobs = [firm for firm in self.model.schedule.agents if isinstance(firm, FirmAgent) and firm.wage > worker.firm.wage and firm.job_openings > 0]
        if higher_paying_jobs:
            return random.choice(higher_paying_jobs)
        else:
            return None
    
    def find_job(self, worker):
        jobs = [firm for firm in self.model.schedule.agents if isinstance(firm, FirmAgent)  and firm.job_openings > 0]
        if jobs:
            return random.choice(jobs)
        else:
            return None

    def should_consider_job_change(self, worker):
        # 自身の生産能力が3以上であれば、20%の確率で転職を考慮
        if worker.production_capacity >= 3 and random.random() <= 0.2:
            return True
        else:
            return False

    def should_seek_job(self, worker):
        # 50%の確率で就職を考慮
        if random.random() <= 0.5:
            return True
        else:
            return False

    def consider_job_change(self):
        # 雇用されている労働者が転職を考慮する
        for worker in self.workers:
            if worker.employed:  # 労働者が現在雇用されている場合
                if self.should_consider_job_change(worker):
                    new_firm = self.find_higher_paying_job(worker)
                    if new_firm is not None and new_firm != worker.firm:
                        worker.firm.fire(worker)  # 以前の雇用者から離職
                        worker.firm = new_firm  # 新しい雇用者に就職
                        worker.employed = True  # 雇用状態を更新

            # 失業している労働者が仕事を探す
            else:
                if self.should_seek_job(worker):
                    new_firm = self.find_job(worker)
                    if new_firm is not None:
                        worker.firm = new_firm  # 新しい雇用者に就職
                        worker.employed = True  # 雇用状態を更新
                        

                    
    # 収入の計算：労働者が働いている企業からの賃金、年金、政府からの社会保障の合計
    def calculate_income(self):
        # 働いている労働者からの賃金の合計
        income_from_wages = sum(worker.firm.wage for worker in self.workers if worker.firm is not None)
        income_from_pensions = self.model.government.pensions(self.num_of_retirees) # 年金受給者からの年金
        income_from_child_allowance = self.model.government.child_allowance(self.num_of_non_workers)  # 児童手当
        income_from_unemployment_allowance = self.model.government.unemployment_allowance(sum(1 for worker in self.workers if worker.firm is None))  # 失業手当
        income_from_BI = self.model.government.BI(self.total_population)  # BI
        
        self.income = income_from_wages + income_from_pensions + income_from_child_allowance + income_from_unemployment_allowance + income_from_BI  # 合計収入
    
    def step(self):
        self.consider_job_change()
        self.calculate_income()
        self.disposable_income = self.income - self.model.government.collect_taxes_house(self.income) #可処分所得=収入-税金
        self.consumption = self.disposable_income * random.uniform(0, 0.8)  # 可処分所得は50%以下を消費
        self.savings += self.disposable_income - self.consumption
        # self.savings += self.model.bank.deposit(self.savings)
        # print(self.savings)


# 企業エージェントのクラス
class FirmAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        # self.capital = 1000 
        self.capital = random.randint(1000, 10000)  # 初期資本
        # self.sales_target = 500
        self.sales_target = random.randint(500, 1000)  # 売上目標
        self.sales = 0  # 売上
        self.average_sales = 0  # 平均売上
        self.profit = 0  # 利益
        self.deficit_period = 0  # 連続赤字期間
        self.hire_workers = []  # 雇用中の労働者リスト
        self.wage = 30  # 初期賃金
        self.debt = 0 #借金
        calculate_capacity = 0 #会社の生産能力
        self.job_openings = 0  # 追加：求人公開数：ここでは不足している生産能力
        self.required_capacity = self.sales_target // 10
        unemployed_workers = [worker for agent in self.model.schedule.agents if isinstance(agent, HouseholdAgent) for worker in agent.workers if not worker.employed]
    
        while self.calculate_total_capacity() < self.required_capacity and unemployed_workers:
            # 雇う: ここでは簡単のため、無職の労働者から最初の人を雇うと仮定します
            worker_to_hire = unemployed_workers.pop(0)
            self.hire(worker_to_hire)
        # print(f"Firm {self.unique_id} has {len(self.hire_workers)} workers.")

    
    def open_job_positions(self):
        """求人を公開する"""
        self.job_openings = self.required_capacity - self.calculate_total_capacity()

    def update_sales(self, total_consumption, total_capacity):
        # 家計エージェントの消費金額を売上に反映する
        if total_capacity > 0:
            self.sales = (self.calculate_total_capacity() / total_capacity) * total_consumption
        else:
            self.sales = 0

    def calculate_total_capacity(self):
        """総生産能力を計算する"""
        self.calculate_capacity = sum(worker.production_capacity for worker in self.hire_workers )
        return self.calculate_capacity
         
    
    def hire_or_fire(self):
        """売上目標に基づいて労働者を雇い、解雇する"""
        while self.calculate_total_capacity() > self.required_capacity and self.hire_workers:
        # 解雇: ここでは生産能力が最も低い労働者を解雇すると仮定します
            worker_to_fire = min(self.hire_workers, key=lambda worker: worker.production_capacity)
            self.fire(worker_to_fire)
            self.open_job_positions()  # 求人を公開する
        self.open_job_positions()  # 求人を公開する
        
    def hire(self, worker):
        # 新たに労働者を雇う
        self.hire_workers.append(worker)
        worker.firm = self
        worker.employed = True

    def fire(self, worker):
    # 労働者を解雇する
        if worker in self.hire_workers:
            self.hire_workers.remove(worker)
            worker.firm = None
            worker.employed = False


    def set_wage(self):
        # 賃金を企業の利益に応じて調整する
        if self.profit > 0 and self.capital > 0 :
            self.wage += 1
        else:
            self.wage -= 1 if self.wage > 1 else 0  # 賃金は1以上

    def calculate_profit(self):
        # 利益を計算する（売上からコスト（賃金）を差し引いたもの）
        self.profit = self.sales - sum(self.wage for _ in self.hire_workers)
        if self.debt == 0 or self.profit < 0 :
            self.capital += self.profit
        else:
            amount_to_repay = self.model.bank.repay(self.profit , self.debt)  # 返済額を計算する
            self.debt -= amount_to_repay
            remaining_amount = self.profit - amount_to_repay  # 返済後の残金を計算する
            if remaining_amount > 0 :
                self.capital += remaining_amount  # 残金がプラスであればそれを資本に加える

    
    def borrowing_decision(self): #借金
        if self.capital < 0 :
            amount_to_borrow = -self.capital  # 借入額はプラスの値でなければならない
            self.debt = self.model.bank.borrow(amount_to_borrow)
            self.capital = 0
    


    def adjust_sales_target(self):
        # 売上目標を調整する
        if self.sales > self.average_sales:
            self.sales_target += 50
        else:
            self.sales_target -= 50 if self.sales_target > 50 else 50  # 売上目標は50以上

    def bankruptcy(self):
        # 倒産する（全ての労働者を解雇する）
        for worker in self.hire_workers:
            self.fire(worker)

    def step(self):
        self.update_sales(self.model.total_consumption, self.model.total_capacity)
        self.calculate_profit()
        self.borrowing_decision()
        self.hire_or_fire()  # 売上目標に基づいて労働者を雇い、解雇する
        self.set_wage()
        self.adjust_sales_target()

        self.sales -= self.model.government.collect_taxes_firm(self.sales)  # 税金を納める

        # 連続赤字期間をカウント
        if self.debt > 0:
            self.deficit_period += 1
            if self.deficit_period >= 12:  # 連続赤字が12期以上続いたら倒産
                self.bankruptcy()
        else:
            self.deficit_period = 0  # 利益が出たら連続赤字期間をリセット
 #       print(f"Firm {self.unique_id} has {len(self.hire_workers)} workers.")



# 政府エージェントのクラス
class GovernmentAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.pension_amount = 0 # 年金
        self.child_allowance_amount = 0  # 児童手当
        self.unemployment_allowance_amount = 0 # 失業手当
        self.BI_amount = 0 # BI
        self.total_amount = 0 #政府の税金での収入と支出．とりあえずマイナスでもいい．
        self.tax_rate_house = 0.4 #家庭への税率
        self.tax_rate_firm = 0.4 #企業への税率
       

    def pensions(self, num_of_retirees):
         # 年金受給者からの年金
         self.total_amount -= self.pension_amount * num_of_retirees
         return self.pension_amount * num_of_retirees

        
    def child_allowance(self,num_of_non_workers) :
        # 児童手当
        self.total_amount -= self.child_allowance_amount * num_of_non_workers
        return self.child_allowance_amount * num_of_non_workers

    def unemployment_allowance(self , num_of_unemployment) :
        # 失業手当
        self.total_amount -= self.unemployment_allowance_amount * num_of_unemployment
        return self.unemployment_allowance_amount * num_of_unemployment
        
    def BI(self, total_population) : 
        # BI
        self.total_amount -= self.BI_amount * total_population
        return self.BI_amount * total_population

    def collect_taxes_house(self, income):
        self.total_amount += self.tax_rate_house * income
        return self.tax_rate_house * income
    
    def collect_taxes_firm(self, sales):
        self.total_amount += self.tax_rate_firm * sales
        return self.tax_rate_firm * sales



    def step(self):
        pass
# 銀行エージェントのクラス
class BankAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.deposits = 0
        self.interest_rate = 0.01


    def deposit(self, savings):
        self.deposits += savings * (1 + self.interest_rate)
        return savings * (1 + self.interest_rate) 

    def borrow(self, amount_to_borrow):
        self.deposits -= amount_to_borrow
        return amount_to_borrow
    
    def repay(self, profit , debt) :
        i = profit - debt * (1 + self.interest_rate)
        if i > 0:
            self.deposits += debt * (1 + self.interest_rate)
            return debt * (1 + self.interest_rate)
        else :
            self.deposits += profit
            return profit
        

    def step(self):
        pass

# シミュレーションモデルのクラス
class EconomyModel(Model):
    def __init__(self, num_households, num_firms):
        self.num_households = num_households
        self.num_firms = num_firms
        self.schedule = RandomActivation(self)


        self.total_capacity = 0
        self.total_consumption = 0

        self.datacollector = DataCollector(
            model_reporters={
                'Average Household Wealth': compute_average_wealth,
                'Median Household Wealth': compute_median_wealth,
                'Average Household income': compute_average_income,
                'Median Household income': compute_median_income
            }
            # model_reporters={"Firm_{}".format(i): lambda m, i=i+self.num_households: len(m.schedule.agents[i].hire_workers) if isinstance(m.schedule.agents[i], FirmAgent) else 0 for i in range(num_firms)}

        )

        # エージェントの初期化
        for i in range(self.num_households):
            household = HouseholdAgent(i, self)
            self.schedule.add(household)

        for i in range(self.num_firms):
            firm = FirmAgent(self.num_households+i, self)
            self.schedule.add(firm)

        self.government = GovernmentAgent(self.num_households + self.num_firms, self)
        self.bank = BankAgent(self.num_households + self.num_firms + 1, self)
        self.schedule.add(self.government)
        self.schedule.add(self.bank)
    
    

    def step(self):
        self.total_capacity = sum([agent.calculate_total_capacity() for agent in self.schedule.agents if isinstance(agent, FirmAgent)])
        self.total_consumption = sum([agent.consumption for agent in self.schedule.agents if isinstance(agent, HouseholdAgent)])
        self.schedule.step()
        self.datacollector.collect(self)
        

    

# 財産の平均値を計算する関数
def compute_average_wealth(model):
    wealths = [agent.savings for agent in model.schedule.agents if isinstance(agent, HouseholdAgent)]
    return statistics.mean(wealths) if wealths else 0

# 財産の中央値を計算する関数
def compute_median_wealth(model):
    wealths = [agent.savings for agent in model.schedule.agents if isinstance(agent, HouseholdAgent)]
    return statistics.median(wealths) if wealths else 0

# 収入の平均値を計算する関数
def compute_average_income(model):
    wealths = [agent.income for agent in model.schedule.agents if isinstance(agent, HouseholdAgent)]
    return statistics.mean(wealths) if wealths else 0

# 収入の中央値を計算する関数
def compute_median_income(model):
    wealths = [agent.income for agent in model.schedule.agents if isinstance(agent, HouseholdAgent)]
    return statistics.median(wealths) if wealths else 0

# 企業ごとの従業員の人数を計算する関数
def compute_firm_worker(model):
    workers = []


# メインの実行部分
num_households = 100
num_firms = 10
num_steps = 100

model = EconomyModel(num_households, num_firms)

for _ in range(num_steps):
    model.step()

data = model.datacollector.get_model_vars_dataframe()

plt.plot(data.index, data['Average Household Wealth'], label='Average Wealth')
plt.plot(data.index, data['Median Household Wealth'], label='Median Wealth')
# plt.plot(data.index, data['Average Household income'], label='Average income')
# plt.plot(data.index, data['Median Household income'], label='Median income')
plt.xlabel('Steps')
plt.ylabel('Wealth')
plt.legend()
plt.show()
# data.plot()
# plt.title('Number of Employees per Firm Over Time')
# plt.ylabel('Number of Employees')
# plt.xlabel('Steps')
# plt.show()
