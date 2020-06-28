// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Eliminating channels with zero filter values
// This pass serves as a compression step after the pruning 
// mask is applied in Pytorch or Tensorflow

// The pass will be applied to following cases
// case 1: Conv + Conv
// case 2: Conv + Activation + Conv
// Case 3: Conv + BN + Activation + Conv
// Case 4: Conv + (BN / Act / Pool) + Matmul


#include <numeric>

#include "onnx/common/assertions.h"
#include "onnx/optimizer/pass.h"
#include "onnx/onnx_pb.h"
#include <queue>
#include <list>
#include <map>

namespace ONNX_NAMESPACE {
namespace optimization {
    struct DeleteOperation {
        Node* src
        int64_t axis;
        std::vector<int64_t> ids;
    };

    struct PruneZeroChannels final : public PredicateBasedPass {
        const int32_t FILTER_OUT_AXIS = 0;
        const int32_t FILTER_IN_AXIS = 1;
        const int32_t FILTER_DIM_NUM = 4;
        const int32_t BN_PARAM_NUM = 4;
        const int32_t MUL_DIM_NUM = 2;
        const double ZERO_THRES = 0.00001;
        std::set<ONNX_NAMESPACE::NodeKind> activation_ops;
        std::map<Node*, DeleteOperation> del_ops; 

        explicit PruneZeroChannels()
            : PredicateBasedPass(
                PassType::Fuse,
                PassEfficiency::Complete,
                PassOptimizationType::Compute){

            activation_ops.insert(kPRelu);
            activation_ops.insert(kLeakyRelu);
            activation_ops.insert(kClip);
            activation_ops.insert(kGlobalAveragePool);
            activation_ops.insert(kPool);
            activation_ops.insert(kAdd);
        }
        std::string getPassName() const override {
            return "prune_zero_channel";
        }

        std::vector<int32_t> find_zero_channels(Node *node, Graph& graph) {
            std::vector<int32_t> ret;

            ONNX_ASSERT(node->kind() == kConv || node->kind() == kMatMul);

            auto conv_inputs = node->inputs();
            auto w_iter = graph.getInitializer(conv_inputs[1]->uniqueName());
            
            if (node->kind() == kConv) {
                fprintf(stderr, "conv dims = %d\n", w_iter->sizes().size());
                ONNX_ASSERT (w_iter->sizes().size() == FILTER_DIM_NUM);
            }

            if (node->kind() == kMatMul) {
                ONNX_ASSERT (w_iter->sizes().size() == MUL_DIM_NUM);
            }
            
            if (w_iter->elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT && 
                w_iter->elem_type() != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
                fprintf(stderr, "Tensor [%s] is of type %d, skip zero channel discovery.", w_iter->name().c_str(), w_iter->elem_type());
            }

            int out_chans = w_iter->sizes()[0];
            std::vector<double> sum = w_iter->abs_sum<double>(FILTER_OUT_AXIS);
            for (std::vector<double>::iterator iter = sum.begin(); iter < sum.end(); iter++) {
                if (fabs(*iter) < ZERO_THRES) {
                    ret.push_back(iter - sum.begin());
                } 
            }

            print_vec("abs_sum", sum);

            return ret;
        }

        bool patternMatchPredicate(Node *node) override {
            return node->kind() == kMatMul || node->kind() == kConv;
        }

        std::vector<Node*> fetch_succ(Node* n, Graph& graph) {
            std::vector<Node*> ret;
            std::queue<Node*, std::list<Node*>> q;
            q.push(n);

            for (;q.size() > 0;) {
                if (q.empty()) {
                    fprintf(stderr, "q empty, quit!\n");
                } else {
                    fprintf(stderr, "q not empty, size = %lld\n", q.size());
                }

                fprintf(stderr, "%d in q!\n", q.size());
                auto n = q.front();
                if (n == NULL) {
                    fprintf(stderr, "NULL ptr in queue!\n");
                    q.pop();
                    continue;
                }
                fprintf(stderr, "name = %s\n", n->name().c_str());

                q.pop();
                fprintf(stderr, "here! %d left!\n", q.size());
                auto outputs = n->outputs();
                fprintf(stderr, "there!\n");
                fprintf(stderr, "there!\n");
                for (auto o : outputs) {
                    if (o == NULL) {
                        fprintf(stderr, "Empty output ptr?\n");
                        continue;
                    }

                    auto uses = o->uses();
                    for (auto u : uses) {
                        auto nn = u.user;
                        if (nn == NULL) {
                            fprintf(stderr, "Empty output node?\n");
                            continue;
                        }

                        fprintf(stderr, "gagaga! output = %s\n", o->uniqueName().c_str());
                        fprintf(stderr, "output node = %s\n", nn->name().c_str());
                        if (nn->kind() == kBatchNormalization) {
                            fprintf(stderr, "batch norm!\n");
                            ret.push_back(nn);
                            q.push(nn);
                        } else if (this->activation_ops.find(nn->kind()) != this->activation_ops.end()) {
                            fprintf(stderr, "activation!\n");
                            q.push(nn);
                        } else if (nn->kind() == kConv || nn->kind() == kMatMul) {
                            fprintf(stderr, "conv!\n");
                            ret.push_back(nn);
                        } else {
                            fprintf(stderr, "Unknown Operation as follower of Conv/Matmul: %s (%s)\n", nn->kind().toString(), nn->name().c_str());
                        }
                        fprintf(stderr, "lalala\n");
                        fflush(stderr);
                    }
                }

                int32_t qsize = q.size();
                fprintf(stderr, "%d left in q!\n", qsize);
                
                if (qsize == 0) {
                    fprintf(stderr, "breaking...\n");
                    break;
                }
            }

            return ret;
        }

        void replace_initializer(Node* node, Graph& graph, int32_t offset, Tensor new_tensor, Value* old_val) {
            Value* new_val = graph.addInitializerAndInput(new_tensor);
            node->replaceInput(offset, new_val);
            if (old_val->uses().size() == 0) {
                graph.eraseInitializerAndInput(old_val);
            }
        }

        bool finalizePass(Graph &graph) override {
            fprintf(stderr, "Finalizing it ...");

            for (auto iter = this->del_ops.begin(); iter != this->del_ops.end(); iter++) {
                Node* n = iter->first;
                if (n->kind() == kBatchNormalization) {

                } else if (n->kind() == kConv) {

                } else if (n->kind() == kMatMul) {

                }
            }
        }

        bool runTransform(Node* n, Graph& graph, NodeDestroyType& destroy_current) {
            fprintf(stderr, "finding for %s... \n", n->name().c_str());
            std::vector<int32_t> zero_channels = this->find_zero_channels(n, graph);
            if (zero_channels.size() == 0) {
                fprintf(stderr, "skip! \n");
                return false;
            }
            fprintf(stderr, "found! \n");

            fprintf(stderr, "find succ...\n");
            std::vector<Node*> succ = this->fetch_succ(n, graph);
            if (succ.size() == 0) {
                fprintf(stderr, "skip! \n");
                return false;
            }
            fprintf(stderr, "found! \n");
            fflush(stderr);
            
            DeleteOperation dop{n, FILTER_OUT_AXIS, zero_channels};
            this->del_ops[n].push_back(dop);

            for (auto nn : succ) {
                auto s_inputs = nn->inputs();
                if (nn->kind() == kBatchNormalization) {
                    DeleteOperation dop{n, FILTER_OUT_AXIS, zero_channels};
                    this->del_ops[nn].push_back(dop);
                } else if (nn->kind() == kConv || nn->kind() == kMatMul) {
                    DeleteOperation dop{n, FILTER_IN_AXIS, zero_channels};
                    this->del_ops[nn].push_back(dop);
                }
            }

            /*auto inputs = n->inputs();
            auto end_iter = graph.initializers().end();
            auto w_iter = graph.getInitializer(inputs[1]->uniqueName());
            Tensor nw = *w_iter;
            fprintf(stderr, "deleting w... \n");
            nw.delete_rows(FILTER_OUT_AXIS, zero_channels);
            fprintf(stderr, "w deleted, replacing... \n");
            replace_initializer(n, graph, 1, nw, inputs[1]);
            fprintf(stderr, "w replaced \n");

            fprintf(stderr, "B Name = %s\n", inputs[2]->uniqueName().c_str());
            auto b_iter = graph.getInitializer(inputs[2]->uniqueName());
            if (b_iter != end_iter) {
                Tensor nw = *b_iter;
                fprintf(stderr, "deleting b [%d dims]... \n", nw.sizes().size());
                nw.delete_rows(FILTER_OUT_AXIS, zero_channels);
                fprintf(stderr, "b deleted, replacing... \n");
                replace_initializer(n, graph, 2, nw, inputs[2]);
                fprintf(stderr, "b replaced \n");
            }

            for (auto nn : succ) {
                auto s_inputs = nn->inputs();
                if (nn->kind() == kBatchNormalization) {
                    for (int32_t i = 1; i <= BN_PARAM_NUM; i++) {
                        auto p_iter = graph.getInitializer(s_inputs[i]->uniqueName());
                        Tensor nw = *p_iter;
                        fprintf(stderr, "deleting BN... \n");
                        nw.delete_rows(FILTER_OUT_AXIS, zero_channels);
                        fprintf(stderr, "BN deleted, replacing... \n");
                        replace_initializer(nn, graph, i, nw, s_inputs[i]);
                        fprintf(stderr, "BN replaced \n");
                    }
                } else if (nn->kind() == kConv || nn->kind() == kMatMul) {
                    auto sw_iter = graph.getInitializer(s_inputs[1]->uniqueName());
                    Tensor nw = *sw_iter;
                    fprintf(stderr, "deleting fc on %s... \n", nn->name().c_str());
                    nw.delete_rows(FILTER_IN_AXIS, zero_channels);
                    fprintf(stderr, "fc deleted, replacing... \n");
                    replace_initializer(nn, graph, 1, nw, s_inputs[1]);
                    fprintf(stderr, "fc replaced \n");
                }
            }*/
        }
    };
}
}