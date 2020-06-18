import graphviz

class Digraph:
    def __init__(self, n):
        '''
        Constructing a directed graph
        Args:
            n (int): number of vertex in the graph
        '''
        self.__num_vertex = n
        self.__adjacent = {_ : [] for _ in range(n)}

    def add_edge(self, start, end):
        '''
        Add a edge to graph
        Args:
            start (int): start vertex
            end (int): end vertex
        '''
        self.__adjacent[start].append(end)
    
    def adj(self, v):
        '''
        Return a list of nodes which are directly connected 
        with vertex v
        Args:
            v (int): vertex
        Returns:
            list[int]: list of vertex
        '''
        return self.__adjacent[v]

    def num_vertex(self):
        '''
        Return the number of vertex in the graph
        Returns:
            int: number of vertex
        '''
        return self.__num_vertex
    
    def reverse(self):
        '''
        Return a reversed graph
        Returns:
            Digraph: a reversed graph
        '''
        dg = Digraph(self.__num_vertex)
        for v in range(self.__num_vertex):
            for w in self.adj(v):
                dg.add_edge(w, v)
        return dg


class FiniteAutomata:
    def __init__(self, reg):
        '''
        Constructing a finite automata based on an 
        input regular expression.
        Args:
            reg (str): regular expression
        '''
        self.reg = self.__reg_preprocess(reg)
        self.g = self.__build_graph(self.reg)
        self.rg = self.g.reverse()

    def __reg_preprocess(self, reg):
        '''
        Pre-process regular expression, seperate the operators and tokens
        Args:
            reg (str): regular expression
        Returns:
            list[str]: a list of legal regular expression element,
            includes tokens and operators, e.g. (dog|cat) will be 
            processed as: [(, dog, |, cat, )]
        '''
        exp = []
        current_ptr = 0
        while current_ptr < len(reg):
            if reg[current_ptr] in ['(', ')', '|', '?']:
                exp.append(reg[current_ptr])
                current_ptr += 1
            else:
                forward_ptr = current_ptr + 1
                if reg[current_ptr] != ' ':
                    while forward_ptr < len(reg) and \
                            reg[forward_ptr] not in ['(', ')', '|', '?', ' ']:
                        forward_ptr += 1
                    exp.append(reg[current_ptr:forward_ptr])
                current_ptr = forward_ptr
        return exp

    def __build_graph(self, reg):
        '''
        Construct finite automata, which is represented as direct graph
        Args:
            reg (list[str]): a processed regular expression
        '''
        g = Digraph(len(reg) + 1)
        stack = []
        ors = []
        cached = None
        for i, c in enumerate(reg):
            if c == '(':
                stack.append(i)
                g.add_edge(i, i + 1)
            elif c == ')':
                assert len(stack) != 0, 'parentheses does not match'
                cached = stack.pop(-1)
                g.add_edge(i, i + 1)
            elif c == '?':
                prev = cached if cached is not None else i - 1
                g.add_edge(i, i + 1)
                g.add_edge(i + 1, prev)
            elif c == '|':
                prev = cached if cached is not None else i - 1
                # g.add_edge(i, i + 1)
                g.add_edge(prev, i + 1)
                ors.append(i)
            else:
                cached = None
                while len(ors) != 0:
                    o = ors.pop(-1)
                    if reg[i - 1] == '|':
                        g.add_edge(o, i + 1)
                    else:
                        g.add_edge(o, i)
        while len(ors) != 0:
            o = ors.pop(-1)
            g.add_edge(o, g.num_vertex() - 1)
        return g

    def __transition(self, v):
        '''
        Return a transition table which indicates what other
        states can transist to state v by consuming some input
        token.
        Args:
            v (int): vertex
        Returns:
            Dict{int: [str]}: a transition table which indicates
            some other states can transist to state v by consuming
            s token in the list, e.g. {1: [animal], 3: [dog]}
            indicates state 1 can transist to v by consuming one
            of the token in the list [animal] and state 3 can
            transist to v by consuming token dog. 
        '''
        table = {}
        queue = [v]
        marked = set()
        ops_tokens = ['(', ')', '|', '?']
        while len(queue) != 0:
            w = queue.pop(0)
            if w not in marked:
                if w > 0:
                    token = self.reg[w - 1]
                    if token not in ops_tokens:
                        table[w - 1] = token
                queue.extend(self.rg.adj(w))
                marked.add(w)
        return table

    def transitions(self):
        '''
        Return the transition table for the entire finite automata
        Returns:
            Dict{int: Dict{int: list[str]}}: Each key in the 
            return dict represent a vertex(state) and it's value
            is the small transition table associate with this vertex,
            i.e. obtained by calling __transition()
        '''
        tables = {}
        for v in range(self.g.num_vertex()):
            tables[v] = self.__transition(v)
        return tables
    
    def visualize(self):
        '''
        Use graphviz to visualize finite automata, the red
        arrow in the figure is eplison-transition.
        '''
        f = graphviz.Digraph('finite_state_machine')
        f.attr(rankdir='LR', size='8,5')
        for v in range(self.g.num_vertex()):
            source = 'S{}'.format(v)
            for w in self.g.adj(v):
                target = 'S{}'.format(w)
                f.edge(source, target, color='red')
        for i in range(len(self.reg)):
            if self.reg[i] not in ['(', ')', '|', '?']:
                source = 'S{}'.format(i)
                target = 'S{}'.format(i + 1)
                f.edge(source, target, label=self.reg[i])
        return f