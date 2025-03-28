#ifndef LP_MODEL
#define LP_MODEL

#include <Eigen/Dense>
#include <optional>
#include <vector>

/**
 * TODO
 *
 * 1. Add objective to the LP model
 * 2. Add method "toStandard" which converts program to standard form
 *      * On this point, consider if it should generate a copy of the LP model
 *      * Perk would be original is still there, con would be expensive.
 *
 */

namespace lp {

struct MatrixContainer {
    std::shared_ptr<Eigen::MatrixXd> A;
    std::shared_ptr<Eigen::MatrixXd> b;
    std::shared_ptr<Eigen::MatrixXd> c;
};

struct ConstraintRows {
    Eigen::MatrixXd lhs;
    Eigen::MatrixXd rhs;
    ConstraintRows() = default;
    ConstraintRows(const Eigen::MatrixXd &lhs_matrix,
                   const Eigen::MatrixXd &rhs_matrix)
        : lhs(lhs_matrix), rhs(rhs_matrix) {}
};

struct LpVariable {
    const size_t id;
    double val;
    std::pair<double, double> bounds = {std::numeric_limits<double>::lowest(),
                                        std::numeric_limits<double>::max()};

    void setLb(const double lb);
    void setUb(const double ub);
    void setBounds(const double lb, const double ub);
    const size_t getId(void) const { return id; };
    const ConstraintRows generateRows(const size_t num_variables) const;
};

/**
 * @brief struct to represent an LP constraint
 * Modeled as follows:
 * vars hold the indices of the variables in the LpSolver
 * coeffs are their corresponding coefficients
 * s.t.
 * lb <= sum_i lp_variables.at(i).val * coeff.at(i) <= ub
 */
struct LpConstraint {
    // first = lb, second = ub
    std::pair<double, double> bounds = {std::numeric_limits<double>::lowest(),
                                        std::numeric_limits<double>::max()};
    std::vector<size_t> vars;
    std::vector<double> coeffs;
    void setBounds(const double lb, const double ub);
    void setLb(const double lb);
    void setUb(const double ub);
    void addVariable(const double coeff, const std::shared_ptr<LpVariable> var);
    const ConstraintRows generateRows(const size_t num_variables) const;
};

class LpSolver {
   private:
    std::vector<std::shared_ptr<LpVariable>> lp_variables_ = {};
    std::vector<std::shared_ptr<LpConstraint>> lp_constraints_ = {};
    std::vector<double> lp_objective_ = {};
    std::vector<std::shared_ptr<LpVariable>> lp_variables_standard_ = {};
    std::vector<std::shared_ptr<LpConstraint>> lp_constraints_standard_ = {};
    std::vector<double> lp_objective_standard_ = {};

   public:
    std::shared_ptr<LpVariable> addVariable();
    std::shared_ptr<LpConstraint> addConstraint();
    const std::vector<std::shared_ptr<LpConstraint>> getConstraints(
        void) const {
        return lp_constraints_;
    }
    void addToObjective(const LpVariable &var, const double coeff);
    const std::vector<std::shared_ptr<LpVariable>> getVars(void) const {
        return lp_variables_;
    }
    // Get matrix representing the constriants
    MatrixContainer getMatrices();
    Eigen::MatrixXd getMatricesStandard();

    const int numConstraintRows() const;
    double evaluateObjective(void);

    void toStandard(void) {};
};

}  // namespace lp

#endif /* LP_MODEL */