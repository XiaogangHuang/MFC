#ifndef DISJSET_H_INCLUDED
#define DISJSET_H_INCLUDED
#include <vector>

class DisjSet
{
  private:
      std::vector<int> Parent;
      std::vector<int> Rank;
      int sz;

  public:
      DisjSet(int max_size) : Parent(std::vector<int>(max_size)),
                              Rank(std::vector<int>(max_size, 0)),
                              sz(max_size)
      {
          for (int i = 0; i < max_size; ++i)
              Parent[i] = i;
      }
      int Size();
      void reSize(int s);
      int Find(int x);
      void Union(int x1, int x2);
      bool Is_same(int e1, int e2);
};

#endif // DISJSET_H_INCLUDED
