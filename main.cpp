#include <iostream>
#include <mpi.h>
#include <vector>
#include <ctime>
#include <random>
#include <map>

/*

Алгоритм в следующем.
У каждого процесса общая область памяти, то есть, то каждый процесс знает, какие общие переменные определены.
При записи в переменную процесс всем через MPI_Isend посылает новое значение переменной, номер переменной передаётся в теге.
Каждой своей записи он присваивает свой номер. Номер нового ожидаемого сообщения хранится у каждого процесса
При чтении процесс с помощью MPI_Irecv с MPI_ANY_SOURCE и с номером переменной в теге принимает сообщение с этой переменной.
Если номер сообщения от процесса, который сделал запись в эту переменную, не равен ожидаемому, процесс продолжает слушать сообщения от этого процесса,
но уже с MPI_ANY_TAG.

Данный алгоритм обеспечивает процессорную консистентность, каждый процесс может одновременно читать и писать, то есть,
это DSM с полным размножением.

*/

namespace {
    using DATATYPE = int;
    constexpr int root_id = 0;
    size_t send_counts = 0;
    MPI_Datatype MPI_DATATYPE;
    constexpr int Ts = 100;
    constexpr int MSG_SIZE = 4;
    size_t bytes_sent = 0;
    constexpr int OK = 1;
}

// Класс общей памяти
template<typename T>
class Shared_Vars {
    public:
        Shared_Vars(const std::vector<std::string>& vars, T init_val = 0); // инициализация
        T read(const std::string& var); // чтение
        void write(const std::string& var, T val); // запись
        void show_vals() const; // показать текущие значения переменных внутри процессов
        void wait_all();
    private:
        void AskAccess (const std::string& var);
        void SendToAll (const std::string& var, T val); // послать всем процессам новое значение переменной
        std::map<int, size_t> vars_versions; // номер изменений переменных внутри процессов
        std::map<std::string, int> hash_var; // перевод чисел из их названий в номера
        std::vector<int> msg_procs; // количество принятых сообщений от каждого процесса
        std::map<int, T> vals; // текущие значения переменных

        std::vector<int> access_buf;
        int CRITICAL_TAG;
        int ANSWER_TAG;
        int END_TAG;
};

template<typename T>
void Shared_Vars<T>::wait_all() {
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Request request;
    MPI_Status status;
    std::vector<int> msg(MSG_SIZE);

    std::cout << rank << ":" << msg_procs[rank] << std::endl;

    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            MPI_Isend(&msg_procs[rank], 1, MPI_INT, i, END_TAG, MPI_COMM_WORLD, &request);
        }
    }

    int msg_proc_count;
    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            MPI_Recv(&msg_proc_count, 1, MPI_INT, i, END_TAG, MPI_COMM_WORLD, &status);

            int cur_msg_num = msg_procs[i];
            while (cur_msg_num < msg_proc_count - 1) {
                std::cout << rank << " " << cur_msg_num << " " << msg_proc_count - 1 << std::endl;
                MPI_Recv(msg.data(), MSG_SIZE, MPI_DATATYPE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                cur_msg_num++;
            }
        }
    }
}

template <typename T>
void Shared_Vars<T>::SendToAll(const std::string& var, T val) {
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Request request;

    int hash_num = hash_var[var];
    std::vector<T> msg(MSG_SIZE);
    msg[0] = msg_procs[rank];
    msg_procs[rank]++;
    msg[1] = val;
    
    vars_versions[hash_num]++;
    msg[2] = vars_versions[hash_num];
    msg[3] = hash_num;

    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            std::cout << rank << " -> " << i << " " <<  hash_num << " " << msg[0] << " " << msg[1] << " " << msg[2] << std::endl;
            send_counts++;
            bytes_sent += MSG_SIZE * sizeof(DATATYPE);
            MPI_Isend(msg.data(), MSG_SIZE, MPI_DATATYPE, i, hash_num, MPI_COMM_WORLD, &request);
        }
    }
}

template <typename T>
void Shared_Vars<T>::AskAccess(const std::string& var) {
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Request request;
    MPI_Status status;

    //std::cout << "WORLD: " << world_size << std::endl;

    int hash_num;
    if (var != "") {
        hash_num = hash_var[var];
    } else {
        hash_num = -1;
    }

    msg_procs[rank]++;

    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            send_counts++;
            bytes_sent += sizeof(int);
            MPI_Isend(&hash_num, 1, MPI_INT, i, CRITICAL_TAG, MPI_COMM_WORLD, &request);
        }
    }

    int msg;
    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            MPI_Recv(&msg, 1, MPI_INT, i, CRITICAL_TAG, MPI_COMM_WORLD, &status);

            if (msg != hash_num or i < rank) {
                send_counts++;
                bytes_sent += sizeof(int);
                MPI_Isend(&OK, 1, MPI_INT, i, ANSWER_TAG, MPI_COMM_WORLD, &request);
            } else {
                access_buf.push_back(i);
            }
        }
    }

    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            MPI_Recv(&msg, 1, MPI_INT, i, ANSWER_TAG, MPI_COMM_WORLD, &status);
        }
    }
}

template<typename T>
Shared_Vars<T>::Shared_Vars(const std::vector<std::string>& vars, T init_val) {
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    msg_procs = std::vector<int>(world_size, 0);
    int var_to_hash = 0;

    for (const auto& var: vars) {
        hash_var[var] = var_to_hash;
        vals[var_to_hash] = init_val;
        vars_versions[var_to_hash] = 0;
        var_to_hash++;
    }

    CRITICAL_TAG = vars.size();
    ANSWER_TAG = CRITICAL_TAG + 1;
    END_TAG = ANSWER_TAG + 1;
}

template<typename T>
T Shared_Vars<T>::read(const std::string& var) {
    std::vector<T> msg(MSG_SIZE);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Request request;
    MPI_Status status;
    int flag;
    int hash_num = hash_var[var];

    int iter_count = 0;    
    do {
        /*
        if (rank == 0) {
            std::cout << "HERE2: " << std::endl;
            for (size_t i = 0; i < access_buf.size(); i++) {
                std::cout << access_buf[i] << " ";
            }
            std::cout << std::endl;
        }
        */
        std::cout << rank << ": READ - " << hash_num << std::endl;
        MPI_Irecv(msg.data(), MSG_SIZE, MPI_DATATYPE, MPI_ANY_SOURCE, hash_num, MPI_COMM_WORLD, &request);
        MPI_Test(&request, &flag, &status);

        /*
        if (rank == 0) {
            std::cout << "HERE3: " << flag << std::endl;
            for (size_t i = 0; i < access_buf.size(); i++) {
                std::cout << access_buf[i] << " ";
            }
            std::cout << std::endl;
        } 
        */
        int source = status.MPI_SOURCE;

        if (flag == 1) {
            std::cout << rank << ": " << status.MPI_SOURCE << " - " << msg[0] << " " << msg[1] << " " << msg[2] << std::endl;
            if (int(msg[0]) < msg_procs[source]) {
                std::map<int, std::vector<T>> buf;
                buf.insert(std::make_pair(int(msg[0]), msg));

                while(buf.size() != 0) {
                    std::vector<T> buf_msg;
                    MPI_Recv(msg.data(), MSG_SIZE, MPI_DATATYPE, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                    int cur_hash_num = status.MPI_TAG;

                    if (int(msg[0]) == msg_procs[source]) {
                        vals[cur_hash_num] = msg[1];
                        msg_procs[source]++;

                        std::vector<int> commands_to_delete = {};
                        for (const auto& p: buf) {
                            if (p.first == msg_procs[source]) {
                                vals[int(p.second[3])] = p.second[1];
                                msg_procs[source]++;
                                commands_to_delete.emplace_back(p.first);
                            } else {
                                break;
                            }
                        }

                        for (auto command_id: commands_to_delete) {
                            buf.erase(command_id);
                        }
                    } else {
                        buf.insert(std::make_pair(int(msg[0]), msg));
                    }
                }
            } else {
                if (vars_versions[hash_num] < msg[2]) {
                    vals[hash_num] = msg[1];
                    vars_versions[hash_num] = msg[2];
                    msg_procs[source]++;
                } else if (vars_versions[hash_num] == msg[2]) {
                    std::cerr << rank << ": " << var << " WRITE ERROR!" << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }
    } while(flag == 1);

    return vals[hash_num];
}

template<typename T>
void Shared_Vars<T>::write(const std::string& var, T val) {
    AskAccess(var);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (var.size() != 0) {
        read(var);
        std::cout << rank << ": READED\n";

        vals[hash_var[var]] = val;
        SendToAll(var, val);

        for (int proc_id: access_buf) {
            //std::cout << rank << ": " << proc_id << std::endl;
            MPI_Request request;
            send_counts++;
            bytes_sent += sizeof(int);
            MPI_Isend(&OK, 1, MPI_INT, proc_id, ANSWER_TAG, MPI_COMM_WORLD, &request);
        }
        
        access_buf.clear();
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

template<typename T>
void Shared_Vars<T>::show_vals() const {
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    for (int i = 0; i < world_size; i++) {
        if (rank == i) {
            std::cout << "---------- " << i << " ----------" << std::endl;
            for (const auto& p: hash_var) {
                std::cout << p.first << " = " << vals.at(p.second) << " | " << vars_versions.at(p.second) << std::endl;
                std::flush(std::cout);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    int rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (typeid(DATATYPE) == typeid(int)) {
        MPI_DATATYPE = MPI_INT;
    } else if(typeid(DATATYPE) == typeid(double)) {
        MPI_DATATYPE = MPI_DOUBLE;
    } else if(typeid(DATATYPE) == typeid(long)) {
        MPI_DATATYPE = MPI_LONG;
    } else {
        if (rank == root_id) std::cerr << "DATATYPE ERROR!" << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::vector<std::string> vars;

    for (int i = 0; i < world_size; i++) {
        char c = 'a' + i;
        std::string var;
        var.push_back(c);
        vars.push_back(var);
    }

    Shared_Vars<DATATYPE> shared_vars(vars);

    std::srand(time(NULL) + rank);

    //shared_vars.write(vars[rank], rank + 2);

    //shared_vars.show_vals();

    //for (int i = 0; i < world_size; i++) {
    //    shared_vars.read(vars[i]);
    //}

    //shared_vars.show_vals();

    vars[1] = "a";
    //vars[2] = "b";

    shared_vars.write(vars[rank], rank + 10);
    shared_vars.show_vals();

    for (int i = 0; i < world_size; i++) {
        char c = 'a' + i;
        std::string var;
        var.push_back(c);
        vars[i] = c;
    }


    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++) {
        shared_vars.read(vars[i]);
    }

    shared_vars.show_vals();
    MPI_Barrier(MPI_COMM_WORLD);


    if (rank == root_id) std::cout << "Time: " << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++) {
        if (rank == i) {
            std::cout << rank << " : " << send_counts * Ts + bytes_sent << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    shared_vars.wait_all();

    MPI_Finalize();
    return 0;
}