# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 19:29:20 2023

@author: yuto
"""

import random
import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
import matplotlib.pyplot as plt
import numpy as np

class HouseholdAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = random.randint(20, 200) #初期財産
        self.income = 0
        self.consumption = 0
        self.savings = 0
        self.unemployment_benefit = 0
        self.pension = 0
        self.basic_income = 0
        self.profits = 0
        
    def step(self):
        self.income = self.model.employment * self.model.wage #収入=雇用率×賃金
        self.consumption = min(self.income, self.wealth) #消費
        self.savings = max(0, self.income - self.consumption) #貯蓄
        self.wealth += self.savings #財産に貯蓄に回した分を入れる
        self.model.total_consumption += self.consumption #全体の消費
        
        if self.income == 0:
            self.unemployment_benefit = 0.2 * self.wealth #失業手当=0.2×財産
            self.wealth += self.unemployment_benefit #財産に失業手当を加える
            self.model.total_unemployment_benefit += self.unemployment_benefit #全体の失業手当
        
        if self.unique_id in self.model.pension_recipients: #年金受給者ならば
            self.pension = self.model.pension #モデルで指定した年金を支給
            self.wealth += self.pension #財産に年金を加える
            self.model.total_pension += self.pension #全体の年金
            
        self.basic_income = self.model.basic_income #モデルで指定したBIを支給
        self.wealth += self.basic_income #財産にBIを加える
        self.model.total_basic_income += self.basic_income #全体のBI
        

class FirmAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.production_target = random.randint(20, 50) #目標生産量．20から50のランダムな整数
        self.production = 0 #実際の生産量
        self.sales = 0 #販売量
        self.goods_inventory=1000 #在庫
        self.wages_paid = 0 #給与支払額
        self.profits = 0 #利益
        self.loan = 0 #借入金額
        self.income = 0 #企業の所得
        self.wealth = 0 #企業の資産
        
    def step(self):
        self.production = min(self.production_target, self.model.employment) #生産量を目標生産量と雇用率の小さいほう
        self.sales = min(self.production, self.goods_inventory) #売上=生産量と目標生産量の小さいほう
        self.goods_inventory -= self.sales #在庫から販売量を引く
        self.wages_paid = self.model.wage * self.production #賃金支払額=賃金×生産量
        self.profits = self.sales - self.wages_paid #利益=売上-賃金支払額
        
        if self.profits > 0: #利益があれば
            self.production_target += 1 #目標生産量を増やす
        elif self.profits < 0: #利益がなければ
            self.production_target -= 1 #目標生産量を減らす
        
        if self.model.employment < self.production_target:
            shortfall = self.production_target - self.model.employment
            shortfall_funds = shortfall * self.model.wage
            if shortfall_funds > self.model.bank.loan_funds:
                shortfall_funds = self.model.bank.loan_funds
            self.model.bank.loan_funds -= shortfall_funds
            self.loan = shortfall_funds
            self.model.employment += shortfall
        
        self.model.goods_sales += self.sales
        self.model.total_profits += self.profits
        self.model.total_wages_paid += self.wages_paid
        

class GovernmentAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.tax_rate = 0.2
        self.social_security_rate = 0.1
        
    def step(self):
        total_income = 0
        total_wealth = 0
        total_consumption = self.model.total_consumption
        total_wages_paid = self.model.total_wages_paid
        total_profits = self.model.total_profits
        total_unemployment_benefit = self.model.total_unemployment_benefit
        total_pension = self.model.total_pension
        total_basic_income = self.model.total_basic_income
        
        # calculate total income and wealth
        for household in self.model.schedule.agents:
            total_income += household.income
            total_wealth += household.wealth
            
        # tax households and firms
        tax_revenue = total_income * self.tax_rate
        total_wealth -= tax_revenue
        
        for firm in self.model.schedule.agents:
            tax_payment = firm.profits * self.tax_rate
            firm.profits -= tax_payment
            total_wealth += tax_payment
            
        # provide social security benefits
        social_security_fund = total_wages_paid * self.social_security_rate
        total_wealth -= social_security_fund
        
        for household in self.model.schedule.agents:
            household.wealth -= self.social_security_rate * household.income
            household.wealth += self.social_security_rate * total_wages_paid
            
        # provide pension benefits
        for household_id in self.model.pension_recipients:
            household = self.model.schedule._agents[household_id]
            household.wealth += total_pension / len(self.model.pension_recipients)
            total_wealth -= total_pension / len(self.model.pension_recipients)
            
        # provide basic income
        for household in self.model.schedule.agents:
            household.wealth += total_basic_income / self.model.num_households
            total_wealth -= total_basic_income / self.model.num_households
            
        # update model wealth
        self.model.wealth = total_wealth

class Bank(Agent):
    def __init__(self, unique_id, model, init_deposit):
        super().__init__(unique_id, model)
        self.deposit = init_deposit
        self.loan_funds = 0

    def step(self):
        self.pay_interest()

    def receive_deposit(self, amount):
        self.deposit += amount

    def provide_loan(self, amount):
        self.deposit -= amount
        return amount

    def receive_repayment(self, amount):
        self.deposit += amount

    def pay_interest(self):
        interest = self.deposit * self.model.interest_rate
        self.deposit += interest




class EconomicModel(Model):
    def __init__(self, num_households=50,num_firmagents=10, wage=10, pension=1000, basic_income=500):
        self.num_households = num_households #世帯数
        self.num_firmagents = num_firmagents #企業数
        #self.goods_inventory = goods_inventory
        self.wage = wage #賃金
        self.pension = pension #年金
        self.basic_income = basic_income #ベーシックインカム
        self.total_consumption = 0 #全体の消費
        self.total_wages_paid = 0 #賃金支払量
        self.total_profits = 0 #全体の利益
        self.total_unemployment_benefit = 0 #全体の失業手当
        self.total_pension = 0 #全体の年金
        self.total_basic_income = 0 #全体のベーシックインカム
        self.schedule = RandomActivation(self) #すべてのエージェントをステップごとに1回、ランダムな順序でアクティブにする
        self.grid = SingleGrid(1, self.num_households, torus=False)
        self.bank = Bank(0, self, 1000000)
        self.government = GovernmentAgent(1, self)
        self.employment = 0
        self.pension_recipients = set()
        self.wealth = 0
        self.goods_sales = 0
        
        # create household agents
        for i in range(self.num_households):
            a = HouseholdAgent(i+2, self)
            self.schedule.add(a)
            self.grid.position_agent(a)
        
        # create firm agents
        for i in range(self.num_firmagents):
            a = FirmAgent(i+2+self.num_households, self)
            self.schedule.add(a)
        
        self.data = {'step': [], 'mean_income': [], 'median_income': []}
    
    def step(self):
        self.wealth = self.bank.deposit
        self.schedule.step()
        self.government.step()
        self.employment = sum([1 for household in self.schedule.agents if household.income > 0])
        self.pension_recipients = {household.unique_id for household in self.schedule.agents if household.wealth > self.pension}
        self.data['step'].append(self.schedule.steps)
        self.data['mean_income'].append(np.mean([household.income for household in self.schedule.agents]))
        self.data['median_income'].append(np.median([household.income for household in self.schedule.agents]))
        
        if self.schedule.steps == 100:
            self.plot_data()
    
    def plot_data(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.data['step'], self.data['mean_income'], label='Mean income')
        ax.plot(self.data['step'], self.data['median_income'], label='Median income')
        ax.set_xlabel('Step')
        ax.set_ylabel('Disposable income')
        ax.legend()
        plt.show()

if __name__ == '__main__':
    model=EconomicModel()
    wealth_data = []  # エージェントの所得を格納するリスト
    for i in range(100):
        model.step()
        wealth_data.append([a.wealth for a in model.schedule.agents])
    # グラフを描画する
    for i in range(len(wealth_data[0])):
        plt.plot([wealth[i] for wealth in wealth_data])
    model.plot_data()
    
    
