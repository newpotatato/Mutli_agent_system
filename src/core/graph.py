"""
Сервис для генерации и управления графом связности брокеров
"""
import random
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.config import GRAPH_PARAMS


class GraphService:
    def __init__(self, num_brokers):
        self.num_brokers = num_brokers
        self.adjacency_matrix = np.zeros((num_brokers, num_brokers))
        self.weights = {}  # b_ij веса между брокерами
        self.neighbors = {i: [] for i in range(num_brokers)}
        self.edge_probability = GRAPH_PARAMS['edge_probability']
        self.min_neighbors = GRAPH_PARAMS['min_neighbors']
        self.max_neighbors = GRAPH_PARAMS['max_neighbors']
        self.weight_decay = GRAPH_PARAMS['weight_decay']
        
        self._generate_initial_graph()
    
    def _generate_initial_graph(self):
        """
        Генерирует начальный случайный граф связности
        """
        # Создаем случайные связи
        for i in range(self.num_brokers):
            for j in range(i + 1, self.num_brokers):
                if random.random() < self.edge_probability:
                    self._add_edge(i, j)
        
        # Убеждаемся, что каждый узел имеет минимальное количество соседей
        self._ensure_connectivity()
        
        # Инициализируем веса
        self._initialize_weights()
    
    def _add_edge(self, i, j):
        """
        Добавляет ребро между узлами i и j
        """
        self.adjacency_matrix[i][j] = 1
        self.adjacency_matrix[j][i] = 1
        
        if j not in self.neighbors[i]:
            self.neighbors[i].append(j)
        if i not in self.neighbors[j]:
            self.neighbors[j].append(i)
    
    def _remove_edge(self, i, j):
        """
        Удаляет ребро между узлами i и j
        """
        self.adjacency_matrix[i][j] = 0
        self.adjacency_matrix[j][i] = 0
        
        if j in self.neighbors[i]:
            self.neighbors[i].remove(j)
        if i in self.neighbors[j]:
            self.neighbors[j].remove(i)
    
    def _ensure_connectivity(self):
        """
        Обеспечивает минимальную связность графа
        """
        for i in range(self.num_brokers):
            while len(self.neighbors[i]) < self.min_neighbors:
                # Находим случайный узел для подключения
                candidates = [j for j in range(self.num_brokers) 
                            if j != i and j not in self.neighbors[i] 
                            and len(self.neighbors[j]) < self.max_neighbors]
                
                if candidates:
                    j = random.choice(candidates)
                    self._add_edge(i, j)
                else:
                    break  # Нет доступных кандидатов
    
    def _initialize_weights(self):
        """
        Инициализирует веса ребер b_ij = 2 / (deg(i) + deg(j))
        """
        # Вычисляем степени узлов
        degrees = [len(self.neighbors[i]) for i in range(self.num_brokers)]
        
        for i in range(self.num_brokers):
            for j in self.neighbors[i]:
                if i < j:  # Избегаем дублирования
                    # Вес ребра согласно спецификации: b_ij = 2 / (deg(i) + deg(j))
                    weight = 2.0 / (degrees[i] + degrees[j]) if (degrees[i] + degrees[j]) > 0 else 0.0
                    self.weights[(i, j)] = weight
                    self.weights[(j, i)] = weight
    
    def get_neighbors(self, broker_id):
        """
        Возвращает список соседей для данного брокера
        """
        return self.neighbors.get(broker_id, [])
    
    def get_weight(self, broker_i, broker_j):
        """
        Возвращает вес ребра между брокерами i и j
        """
        return self.weights.get((broker_i, broker_j), 0.0)
    
    def update_graph(self):
        """
        Периодически обновляет граф связности
        """
        # Применяем затухание к весам
        for edge, weight in self.weights.items():
            self.weights[edge] = weight * self.weight_decay
        
        # Случайно добавляем или удаляем ребра
        if random.random() < 0.1:  # 10% шанс изменения
            self._random_graph_modification()
    
    def _random_graph_modification(self):
        """
        Случайное изменение структуры графа
        """
        action = random.choice(['add', 'remove', 'modify_weight'])
        
        if action == 'add':
            self._add_random_edge()
        elif action == 'remove':
            self._remove_random_edge()
        elif action == 'modify_weight':
            self._modify_random_weight()
    
    def _add_random_edge(self):
        """
        Добавляет случайное ребро
        """
        candidates = []
        for i in range(self.num_brokers):
            for j in range(i + 1, self.num_brokers):
                if (self.adjacency_matrix[i][j] == 0 and 
                    len(self.neighbors[i]) < self.max_neighbors and
                    len(self.neighbors[j]) < self.max_neighbors):
                    candidates.append((i, j))
        
        if candidates:
            i, j = random.choice(candidates)
            self._add_edge(i, j)
            weight = random.uniform(0.1, 1.0)
            self.weights[(i, j)] = weight
            self.weights[(j, i)] = weight
    
    def _remove_random_edge(self):
        """
        Удаляет случайное ребро (с ограничениями)
        """
        candidates = []
        for i in range(self.num_brokers):
            if len(self.neighbors[i]) > self.min_neighbors:
                for j in self.neighbors[i]:
                    if len(self.neighbors[j]) > self.min_neighbors:
                        candidates.append((i, j))
        
        if candidates:
            i, j = random.choice(candidates)
            self._remove_edge(i, j)
            if (i, j) in self.weights:
                del self.weights[(i, j)]
            if (j, i) in self.weights:
                del self.weights[(j, i)]
    
    def _modify_random_weight(self):
        """
        Изменяет случайный вес ребра
        """
        if self.weights:
            edge = random.choice(list(self.weights.keys()))
            new_weight = random.uniform(0.1, 1.0)
            self.weights[edge] = new_weight
            # Обновляем симметричный вес
            reverse_edge = (edge[1], edge[0])
            if reverse_edge in self.weights:
                self.weights[reverse_edge] = new_weight
    
    def get_graph_stats(self):
        """
        Возвращает статистику графа
        """
        total_edges = sum(len(neighbors) for neighbors in self.neighbors.values()) // 2
        avg_degree = total_edges * 2 / self.num_brokers
        
        return {
            'num_nodes': self.num_brokers,
            'num_edges': total_edges,
            'average_degree': avg_degree,
            'density': total_edges / (self.num_brokers * (self.num_brokers - 1) / 2),
            'weights_count': len(self.weights)
        }
    
    def consensus_update(self, brokers, gamma_consensus=None):
        """
        Выполняет консенсусное SPSA обновление согласно спецификации:
        θ_i ← θ_i - α * (ĝ + γ * ∑_{j∈neighbors[i]} bᵢⱼ * (θ_j - θ_i))
        
        Args:
            brokers: Список или словарь брокеров
            gamma_consensus: Коэффициент консенсуса (по умолчанию из конфига)
        """
        # Если brokers - список, конвертируем в словарь
        if isinstance(brokers, list):
            broker_dict = {broker.id: broker for broker in brokers}
        else:
            broker_dict = brokers
        
        # Используем константы из конфига или заданные
        from configs.config import SPSA_PARAMS
        if gamma_consensus is None:
            gamma_consensus = SPSA_PARAMS.get('gamma_consensus', 0.02)
        
        # Консенсусное обновление согласно спецификации SPSA:
        # θ_i ← θ_i - α * (ĝ + γ * ∑_{j∈neighbors[i]} bᵢⱼ * (θ_j - θ_i))
        
        for broker_id, broker in broker_dict.items():
            neighbors = self.get_neighbors(broker_id)
            
            if not neighbors:
                continue  # Пропускаем брокеров без соседей
            
            # Преобразуем theta в numpy array для корректных вычислений
            theta_i = np.array(broker.theta)
            
            # Вычисляем консенсусный терм: ∑_{j∈neighbors[i]} bᵢⱼ * (θ_j - θ_i)
            consensus_term = np.zeros_like(theta_i)
            
            for neighbor_id in neighbors:
                if neighbor_id in broker_dict:
                    neighbor_theta = np.array(broker_dict[neighbor_id].theta)
                    edge_weight = self.get_weight(broker_id, neighbor_id)
                    consensus_term += edge_weight * (neighbor_theta - theta_i)
            
            # Применяем консенсусное обновление с коэффициентом gamma_consensus
            # Обратите внимание: градиент ĝ уже применен в update_parameters(), 
            # здесь мы только добавляем консенсусную коррекцию
            consensus_correction = gamma_consensus * consensus_term
            
            # Обновляем theta брокера
            updated_theta = theta_i + consensus_correction
            broker.theta = updated_theta.tolist()
        
        # Обновляем веса ребер на основе схожести параметров
        self._update_edge_weights_by_similarity(broker_dict)
    
    def _update_edge_weights_by_similarity(self, broker_dict):
        """
        Обновляет веса ребер на основе схожести параметров theta соседних брокеров
        """
        for edge in list(self.weights.keys()):
            broker_i, broker_j = edge
            
            if broker_i in broker_dict and broker_j in broker_dict:
                # Вычисляем косинусное сходство между theta параметрами
                # Преобразуем theta в numpy arrays для корректных операций
                theta_i = np.array(broker_dict[broker_i].theta)
                theta_j = np.array(broker_dict[broker_j].theta)
                
                # Косинусное сходство
                norm_i = np.linalg.norm(theta_i)
                norm_j = np.linalg.norm(theta_j)
                
                if norm_i > 0 and norm_j > 0:
                    similarity = np.dot(theta_i, theta_j) / (norm_i * norm_j)
                    # Преобразуем сходство в вес (0.1 - 1.0)
                    new_weight = 0.1 + 0.9 * max(0, similarity)
                else:
                    new_weight = 0.5  # Нейтральный вес если один из векторов нулевой
                
                # Обновляем вес с небольшим сглаживанием
                current_weight = self.weights[edge]
                self.weights[edge] = 0.8 * current_weight + 0.2 * new_weight

    def get_adjacency_matrix(self):
        """
        Возвращает матрицу смежности графа
        """
        return self.adjacency_matrix
    
    def visualize_graph(self):
        """
        Выводит простое текстовое представление графа
        """
        print("=== ГРАФ СВЯЗНОСТИ БРОКЕРОВ ===")
        for i in range(self.num_brokers):
            neighbors_info = []
            for j in self.neighbors[i]:
                weight = self.get_weight(i, j)
                neighbors_info.append(f"{j}({weight:.2f})")
            
            print(f"Брокер {i}: {', '.join(neighbors_info) if neighbors_info else 'нет соседей'}")
        
        stats = self.get_graph_stats()
        print(f"\nСтатистика: {stats['num_edges']} ребер, "
              f"плотность {stats['density']:.2f}, "
              f"средняя степень {stats['average_degree']:.2f}")
