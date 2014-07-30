#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>
#include <locale>
#include <iomanip>

std::vector<std::string> split(const std::string& s, const std::string& delim) {
    std::vector<std::string> result;
    if (delim.empty()) {
        result.push_back(s);
        return result;
    }
    std::string::const_iterator substart = s.begin();
    std::string::const_iterator subend;
    while (true) {
        subend = std::search(substart, s.end(), delim.begin(), delim.end());
        std::string temp(substart, subend);
        if (!temp.empty()) {
            result.push_back(temp);
        }
        if (subend == s.end()) {
            break;
        }
        substart = subend + delim.size();
    }
    return result;
}

static bool is_not_alnum(char c)
{
    return !std::isalnum(c);
}

int main(int argc, const char** argv)
{
    // check for correct command line arguments
    if(argc < 3 || argc > 4){
        std::cout << "Usage: " << argv[0] << " infile outfile [namespace]" << std::endl;
        return 2;
    }

    // parse command line
    const char* iname = argv[1];
    const char* oname = argv[2];
    const char* ns = ((argc >= 4)?(argv[3]):"");
    
    // calculate the char array variable name
    std::string varname(iname);
    std::replace_if(varname.begin(), varname.end(), is_not_alnum, '_');

    // calculate the namespaces
    std::vector<std::string> namespaces = split(ns, "::");
/*    std::stringstream nss(ns);
    std::vector<std::string> namespaces;
    std::string ns_part;
    while(std::getline(nss, ns_part, ':')){
        namespaces.push_back(ns_part);
    }
*/
    // open input file
    std::ifstream ifile(iname, std::ios::in | std::ios::binary);
    if(!ifile.is_open()){
        std::cerr << "Unable to open file '" << iname << "'!" << std::endl;
        return 1;
    }

    // open output file
    std::ofstream ofile(oname, std::ios::out | std::ios::binary);
    if(!ofile.is_open()){
        std::cerr << "Unable to write to file '" << oname << "'!" << std::endl;
        return 1;
    }

    // open namespace brackets
    for(int i = 0; i < namespaces.size(); i++)
    {
        ofile << "namespace " << namespaces[i] << "{" << std::endl;
    }

    // write all bytes to the array
    ofile << "extern const char " << varname << "[] = \"";
    unsigned long numchars = 0;
    unsigned char c = ifile.get();
    while(ifile.good()){
        if(numchars % 20 == 0){
            ofile << "\"\n    \"";
        }
        numchars++;

        ofile << "\\x" << std::hex << std::setw(2) << std::setfill('0')
              << (int)c;
        c = ifile.get();
    }
    ofile << "\";" << std::endl;

    // write the array length
    ofile << "extern const unsigned long " << varname << "_len = " << std::dec << std::setw(0)
          << std::setfill(' ') << numchars << ";" << std::endl;

        
    // close namespace brackets
    for(int i = 0; i < namespaces.size(); i++)
    {
        ofile << "}" << std::endl;
    };

    // close files, return success
    ofile.close();
    ifile.close();
    return 0;
}
