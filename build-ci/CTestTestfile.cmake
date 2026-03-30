# CMake generated Testfile for 
# Source directory: C:/Users/Jai Ansh Bindra/TinyGNN
# Build directory: C:/Users/Jai Ansh Bindra/TinyGNN/build-ci
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(TensorTests "C:/Users/Jai Ansh Bindra/TinyGNN/build-ci/test_tensor.exe")
set_tests_properties(TensorTests PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Jai Ansh Bindra/TinyGNN/CMakeLists.txt;91;add_test;C:/Users/Jai Ansh Bindra/TinyGNN/CMakeLists.txt;0;")
add_test(GraphLoaderTests "C:/Users/Jai Ansh Bindra/TinyGNN/build-ci/test_graph_loader.exe")
set_tests_properties(GraphLoaderTests PROPERTIES  TIMEOUT "300" _BACKTRACE_TRIPLES "C:/Users/Jai Ansh Bindra/TinyGNN/CMakeLists.txt;95;add_test;C:/Users/Jai Ansh Bindra/TinyGNN/CMakeLists.txt;0;")
add_test(MatmulTests "C:/Users/Jai Ansh Bindra/TinyGNN/build-ci/test_matmul.exe")
set_tests_properties(MatmulTests PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Jai Ansh Bindra/TinyGNN/CMakeLists.txt;102;add_test;C:/Users/Jai Ansh Bindra/TinyGNN/CMakeLists.txt;0;")
add_test(SpmmTests "C:/Users/Jai Ansh Bindra/TinyGNN/build-ci/test_spmm.exe")
set_tests_properties(SpmmTests PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Jai Ansh Bindra/TinyGNN/CMakeLists.txt;106;add_test;C:/Users/Jai Ansh Bindra/TinyGNN/CMakeLists.txt;0;")
add_test(ActivationsTests "C:/Users/Jai Ansh Bindra/TinyGNN/build-ci/test_activations.exe")
set_tests_properties(ActivationsTests PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Jai Ansh Bindra/TinyGNN/CMakeLists.txt;110;add_test;C:/Users/Jai Ansh Bindra/TinyGNN/CMakeLists.txt;0;")
add_test(GCNTests "C:/Users/Jai Ansh Bindra/TinyGNN/build-ci/test_gcn.exe")
set_tests_properties(GCNTests PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Jai Ansh Bindra/TinyGNN/CMakeLists.txt;114;add_test;C:/Users/Jai Ansh Bindra/TinyGNN/CMakeLists.txt;0;")
add_test(GraphSAGETests "C:/Users/Jai Ansh Bindra/TinyGNN/build-ci/test_graphsage.exe")
set_tests_properties(GraphSAGETests PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Jai Ansh Bindra/TinyGNN/CMakeLists.txt;118;add_test;C:/Users/Jai Ansh Bindra/TinyGNN/CMakeLists.txt;0;")
add_test(GATTests "C:/Users/Jai Ansh Bindra/TinyGNN/build-ci/test_gat.exe")
set_tests_properties(GATTests PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Jai Ansh Bindra/TinyGNN/CMakeLists.txt;122;add_test;C:/Users/Jai Ansh Bindra/TinyGNN/CMakeLists.txt;0;")
add_test(E2ETests "C:/Users/Jai Ansh Bindra/TinyGNN/build-ci/test_e2e.exe")
set_tests_properties(E2ETests PROPERTIES  WORKING_DIRECTORY "C:/Users/Jai Ansh Bindra/TinyGNN" _BACKTRACE_TRIPLES "C:/Users/Jai Ansh Bindra/TinyGNN/CMakeLists.txt;126;add_test;C:/Users/Jai Ansh Bindra/TinyGNN/CMakeLists.txt;0;")
