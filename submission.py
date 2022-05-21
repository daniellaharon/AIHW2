import time

from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random
import signal


class AgentGreedyImproved(AgentGreedy):
    # TODO: section a : 3
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        children_heuristics = [self.heuristic(child, agent_id) for child in children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        other_taxi = env.get_taxi((taxi_id+1) % 2)
        cash = taxi.cash - other_taxi.cash
        fuel = taxi.fuel - other_taxi.fuel
        taxi_distance = 0
        other_taxi_distance = 0
        closest_passenger = 0
        if taxi.passenger is not None:
            taxi_distance = manhattan_distance(taxi.passenger.position,taxi.passenger.destination)
        else:
            closest_passenger = 0
            if len(env.passengers):
                closest_passenger = 16
                for passenger in env.passengers:
                    passenger_distance = manhattan_distance(passenger.position,passenger.destination)
                    closest_passenger = min(closest_passenger,passenger_distance)#find the closest passenger to me
        if other_taxi.passenger is not None:
            other_taxi_distance = manhattan_distance(other_taxi.passenger.position,other_taxi.passenger.destination)
        distance = taxi_distance - other_taxi_distance
        if closest_passenger:
            return cash + fuel + distance + 1/closest_passenger
        return cash + fuel + distance


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        start_time = time.time()
        depth = 0
        max_depth = 100
        val = None
        while depth <= max_depth:
            try:
                val = self.run_minimax_step(env, agent_id, depth, True,start_time,time_limit)
            except TimeoutError:
                break
            depth += 1
        return val

    def run_minimax_step(self, env: TaxiEnv, agent_id, depth, maximizer,start_time,time_limit):
        timeout(start_time, time_limit)
        if env.done or depth == 0:
            return self.heuristic(env, agent_id)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        if maximizer:
            cur_max = float('-inf')
            for child, op in zip(children, operators):
                timeout(start_time, time_limit)
                child.apply_operator(agent_id, op)
                v = self.run_minimax_step(child, 1-agent_id, depth-1, False,start_time,time_limit)
                cur_max = max(v, cur_max)
            return cur_max
        else:
            cur_min = float('inf')
            for child, op in zip(children, operators):
                timeout(start_time, time_limit)
                child.apply_operator(agent_id, op)
                v = self.run_minimax_step(child, 1-agent_id, depth-1, True,start_time,time_limit)
                cur_min = min(v, cur_min)
            return cur_min

    def heuristic(self, env: TaxiEnv, agent_id):
        if env.done:
            return 1
        return 0



class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        start_time=time.time()
        depth = 0
        alpha = float('-inf')
        beta = float('inf')
        max_depth = 100
        val = None
        while depth <= max_depth:
            try:
                val = self.run_AlphaBeta_step(env, agent_id, depth, True, alpha, beta,start_time,time_limit)
            except TimeoutError:
                break
            depth += 1
        return val

    def run_AlphaBeta_step(self, env: TaxiEnv, agent_id, depth, maximizer, alpha, beta,start_time,time_limit):
        timeout(start_time,time_limit)
        if env.done or depth == 0:
            return self.heuristic(env, agent_id)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        if maximizer:
            cur_max = float('-inf')
            for child, op in zip(children, operators):
                timeout(start_time, time_limit)
                child.apply_operator(agent_id, op)
                v = self.run_AlphaBeta_step(child, 1-agent_id, depth-1, False, alpha, beta,start_time,time_limit)
                cur_max = max(v, cur_max)
                alpha = max(cur_max, alpha)
                if cur_max >= beta:
                    return float('inf')
            return cur_max
        else:
            cur_min = float('inf')
            for child, op in zip(children, operators):
                timeout(start_time, time_limit)
                child.apply_operator(agent_id, op)
                v = self.run_AlphaBeta_step(child, 1-agent_id, depth-1, True, alpha, beta)
                cur_min = min(v, cur_min)
                beta = min(cur_min, beta)
                if cur_min <= alpha:
                    return float('-inf')
            return cur_min


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        start_time = time.time()
        depth = 0
        max_depth = 100
        val = None
        while depth <= max_depth:
            try:
                val = self.run_expectimax_step(env, agent_id, depth, True,start_time,time_limit)
            except TimeoutError:
                break
            depth += 1
        return val

    def run_expectimax_step(self, env: TaxiEnv, agent_id, depth, maximizer,start_time,time_limit):
        timeout(start_time, time_limit)
        if env.done or depth == 0:
            return self.heuristic(env, agent_id)
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        if maximizer:
            cur_max = float('-inf')
            for child, op in zip(children, operators):
                timeout(start_time, time_limit)
                child.apply_operator(agent_id, op)
                v = self.run_expectimax_step(child, 1-agent_id, depth-1, False,start_time,time_limit)
                cur_max = max(v, cur_max)
            return cur_max
        else:
            cur_min = float('inf')
            for child, op in zip(children, operators):
                timeout(start_time, time_limit)
                child.apply_operator(agent_id, op)
                v = self.run_expectimax_step(child, 1-agent_id, depth-1, True,start_time,time_limit)
                cur_min = min(v, cur_min)
            return cur_min

def timeout(start_time,limit_time):
    if time.time()-start_time>=limit_time:
        raise TimeoutError