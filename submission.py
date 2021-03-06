import time
from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance

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
        if taxi.passenger is not None:
            distance = 16-manhattan_distance(taxi.position,taxi.passenger.destination)
            return taxi.cash*16 + distance + taxi.fuel
        else:
            closest_passenger = 0
            if len(env.passengers):
                closest_passenger = 16
                for passenger in env.passengers:
                    passenger_distance = manhattan_distance(taxi.position, passenger.position)
                    closest_passenger = min(closest_passenger, passenger_distance)  # find the closest passenger to me
            return taxi.cash*16 - closest_passenger + taxi.fuel


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        start_time = time.time()
        depth = 1
        move = env.get_legal_operators(agent_id)[0]
        prev_move = move
        while depth:
            if not time.time() - start_time >= time_limit*0.88:
                prev_move=move
                _,move = self.run_minimax_step(env, agent_id, depth, True,start_time,time_limit)
                if move==None:
                    return prev_move
            else:
                return move
            depth += 1
        return move

    def run_minimax_step(self, env: TaxiEnv, agent_id, depth, maximizer, start_time, time_limit):
        if time.time() - start_time >= time_limit*0.88:
            return 0,None
        if env.done() or depth == 0:
            return self.heuristic(env, agent_id), env.get_legal_operators(agent_id)[0]
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        if maximizer:
            cur_max = float('-inf')
            return_op = env.get_legal_operators(agent_id)[0]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                v, _ = self.run_minimax_step(child, 1-agent_id, depth-1, not maximizer, start_time, time_limit)
                if v >= cur_max:
                    return_op = op
                cur_max = max(v, cur_max)
            return cur_max, return_op
        else:
            cur_min = float('inf')
            return_op = env.get_legal_operators(agent_id)[0]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                v, _ = self.run_minimax_step(child, 1-agent_id, depth-1, not maximizer, start_time, time_limit)
                if v <= cur_min:
                    return_op = op
                cur_min = min(v, cur_min)
            return cur_min, return_op

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        if taxi.passenger is not None:
            distance = 16-manhattan_distance(taxi.position,taxi.passenger.destination)
            return taxi.cash*16 + distance + taxi.fuel
        else:
            closest_passenger = 0
            if len(env.passengers):
                closest_passenger = 16
                for passenger in env.passengers:
                    passenger_distance = manhattan_distance(taxi.position, passenger.position)
                    closest_passenger = min(closest_passenger, passenger_distance)  # find the closest passenger to me
            return taxi.cash*16 - closest_passenger + taxi.fuel

class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        start_time=time.time()
        depth = 1
        alpha = float('-inf')
        beta = float('inf')
        move = env.get_legal_operators(agent_id)[0]
        prev_move = move
        while depth:
            if not time.time() - start_time >= time_limit * 0.88:
                prev_move=move
                _,move = self.run_AlphaBeta_step(env, agent_id, depth, True, alpha, beta, start_time, time_limit)
                if move==None:
                    return prev_move
            else:
                return move
            depth += 1
        return move


    def run_AlphaBeta_step(self, env: TaxiEnv, agent_id, depth, maximizer, alpha, beta, start_time, time_limit):
        if time.time() - start_time >= time_limit*0.88:
            return 0,None
        if env.done() or depth == 0:
            return self.heuristic(env, agent_id),env.get_legal_operators(agent_id)[0]
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        if maximizer:
            cur_max = float('-inf')
            return_op = env.get_legal_operators(agent_id)[0]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                v ,_= self.run_AlphaBeta_step(child, 1-agent_id, depth-1, False, alpha, beta, start_time, time_limit)
                if v >= cur_max:
                    return_op = op
                cur_max = max(v, cur_max)
                alpha = max(cur_max, alpha)
                if cur_max >= beta:
                    return float('inf'),op
            return cur_max,return_op
        else:
            cur_min = float('inf')
            return_op = env.get_legal_operators(agent_id)[0]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                v,_ = self.run_AlphaBeta_step(child, 1-agent_id, depth-1, True, alpha, beta, start_time, time_limit)
                if v <= cur_min:
                    return_op = op
                cur_min = min(v, cur_min)
                beta = min(cur_min, beta)
                if cur_min <= alpha:
                    return float('-inf'), op
            return cur_min , return_op

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        if taxi.passenger is not None:
            distance = 16-manhattan_distance(taxi.position,taxi.passenger.destination)
            return taxi.cash*16 + distance + taxi.fuel
        else:
            closest_passenger = 0
            if len(env.passengers):
                closest_passenger = 16
                for passenger in env.passengers:
                    passenger_distance = manhattan_distance(taxi.position, passenger.position)
                    closest_passenger = min(closest_passenger, passenger_distance)  # find the closest passenger to me
            return taxi.cash*16 - closest_passenger + taxi.fuel

class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        start_time = time.time()
        depth = 1
        move = env.get_legal_operators(agent_id)[0]
        prev_move = move
        while depth:
            if not time.time() - start_time >= time_limit * 0.88:
                prev_move=move
                _, move = self.run_expectimax_step(env, agent_id, depth, True, start_time, time_limit)
                if move==None:
                    return prev_move
            else:
                return move
            depth += 1
        return move

    def run_expectimax_step(self, env: TaxiEnv, agent_id, depth, maximizer,start_time,time_limit):
        if time.time() - start_time >= time_limit*0.88:
            return 0,None
        if env.done() or depth == 0 :
            return self.heuristic(env, agent_id), env.get_legal_operators(agent_id)[0]
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        if maximizer:
            cur_max = float('-inf')
            return_op = env.get_legal_operators(agent_id)[0]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                v, _ = self.run_expectimax_step(child, 1-agent_id, depth-1, not maximizer, start_time, time_limit)
                if v >= cur_max:
                    return_op = op
                cur_max = max(v, cur_max)
            return cur_max, return_op
        else:
            probs = self.getProbs(operators)
            v = 0
            final_res = 0
            cur_max = float('-inf')
            return_op = env.get_legal_operators(agent_id)[0]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                v, _= self.run_expectimax_step(child, 1 - agent_id, depth - 1, not maximizer, start_time, time_limit)
                final_res += probs[op]*v
                if v >= cur_max:
                    return_op = op
                cur_max = max(v, cur_max)
            return final_res, return_op

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        if taxi.passenger is not None:
            distance = 16-manhattan_distance(taxi.position,taxi.passenger.destination)
            return taxi.cash*16 + distance + taxi.fuel
        else:
            closest_passenger = 0
            if len(env.passengers):
                closest_passenger = 16
                for passenger in env.passengers:
                    passenger_distance = manhattan_distance(taxi.position, passenger.position)
                    closest_passenger = min(closest_passenger, passenger_distance)  # find the closest passenger to me
            return taxi.cash*16 - closest_passenger + taxi.fuel

    def getProbs(self,operators):
        sum = 0
        dict = {}
        for op in operators:
            dict[op] = self.isMovement(op)
            sum+=dict[op]
        prob = 1/sum
        for op in operators:
            dict[op] = dict[op]*prob
        return dict

    def isMovement(self,op):
        if op =='move north' or op=='move south' or op == 'move west' or op =='move east':
            return 1
        else:
            return 2
