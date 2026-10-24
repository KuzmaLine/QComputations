/**
 * \mainpage QComputations
 *
 * Квантово-механическая библиотека для симуляции: временно-зависимые
 * гамильтонианы, квантовые состояния и матрицы. Бэкенды: CPU (Intel
 * oneAPI/MKL), CUDA, MPI.
 *
 * \section{intro Введение}
 * Страница сгенерирована автоматически Doxygen. Полный референс API
 * собирается из дерева исходников \c src/QComputations.
 *
 * Сейчас в исходниках практически нет док-аннотаций (\c @brief), поэтому
 * включено \c EXTRACT_ALL: доки строятся для всех символов, но без
 * описаний. Добавляй \c @brief/\c @param/\c @return — см. \ref docguide.
 *
 * \section{core Ключевые сущности}
 * - \c Hamiltonian        — временно-зависимый гамильтониан и его диагонализация
 * - \c TCH_State          — квантовое состояние на вычислительной сетке
 * - \c Matrix, \c CSR_Matrix, \c Blocked_Matrix / \c Blocked_Vector — бэкенды матриц
 * - \c Quantum_Operators, \c Additional_Operators — операторы и наблюдаемые
 * - \c Big_Uint, \c Graph — вспомогательные структуры
 * - CUDA / MPI модули для больших систем
 *
 * \section{backends Варианты сборки (build variants)
 * \verbatim
 *   single         - CPU, Intel oneAPI (базовый)
 *   cluster        - + MPI / кластер
 *   cuda           - CUDA-ядро (__CUDACC__)
 *   blocked_cuda   - blocked CUDA
 * \endverbatim
 *
 * \section{docguide Как добавлять описания}
 * \verbatim
 * class MyClass {
 *  public:
 *   /// Краткое описание метода. (одна строка -> "brief")
 *   void do_something(int x);
 * };
 * \endverbatim
 * Полные правила Doxygen-комментариев: \c @brief, \c @param, \c @return,
 * \c @tparam, \c @note, \c @see, \c @sa, \c @pre, \c @post, \c \c[in],
 * \c \c[out], \c \c[in,out]. Блок доки ставится сразу перед объявлением.
 */
