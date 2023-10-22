#include "DisjSet.h"


int DisjSet::Size()
{
    return this->sz; 
}

void DisjSet::reSize(int s)
{
    this->sz = s;
    Parent.resize(s);
    Rank.resize(s);
    for (int i = 0; i < s; ++i)
    {
        Parent[i] = i;
        Rank[i] = 1;
    }

}

int DisjSet::Find(int x)
{
    return x == Parent[x] ? x : (Parent[x] = Find(Parent[x]));
}

void DisjSet::Union(int x1, int x2)
{
    int f1 = Find(x1);
    int f2 = Find(x2);
    if (f1 == f2)
        return;
    if (Rank[f1] > Rank[f2])
    {
        Parent[f2] = f1;
    }
    else
    {
        Parent[f1] = f2;
        if (Rank[f1] == Rank[f2])
            ++Rank[f2];
    }
    --sz;
}

bool DisjSet::Is_same(int e1, int e2)
{
    return Find(e1) == Find(e2);
}

