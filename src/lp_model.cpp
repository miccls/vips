// std
#include <format>
#include <iostream>
#include <numeric>
#include <ranges>
#include <vector>

// Eigen
#include <Eigen/Dense>

// src
#include "lp_model.h"
#include "utils.h"

/**
 *
 * I want to have the logic of converting the LP to
 * standard form separated.
 * Maybe we should just have variables and constraints with bounds?
 *
 * Order of operation would at least, probably, be to first introduce new
 * variables such that all are non-negative.
 *
 *
 */

namespace {

void validateBounds(const double lb, const double ub) {
    if (ub < lb) {
        throw std::invalid_argument(
            "Upper bound must be larger than lower bound.");
    }
}

void commonSetBounds(std::pair<double, double>& bounds,
                     const std::optional<double> lb,
                     const std::optional<double> ub) {
    validateBounds(lb.value_or(bounds.first), ub.value_or(bounds.second));
    if (lb.has_value()) {
        bounds.first = lb.value();
    }
    if (ub.has_value()) {
        bounds.second = ub.value();
    }
}

const lp::ConstraintRows commonGenerateRows(
    const Eigen::MatrixXd& row, const std::pair<double, double> bounds) {
    const size_t num_variables = row.cols();
    bool bound_below = bounds.first != std::numeric_limits<double>::lowest();
    bool bound_above = bounds.second != std::numeric_limits<double>::max();

    if (bound_above && bound_below) {
        Eigen::MatrixXd A(2, num_variables);
        Eigen::MatrixXd b(2, 1);
        A << row, (row * -1);
        b << bounds.first, -bounds.second;
        return lp::ConstraintRows(A, b);
    }

    Eigen::MatrixXd b(1, 1);

    if (bound_below) {
        b << bounds.first;
        return lp::ConstraintRows(row, b);
    } else {
        b << bounds.second;
        return lp::ConstraintRows(-row, -b);
    }
}

}  // namespace

// Lp solver

std::shared_ptr<lp::LpVariable> lp::LpSolver::addVariable() {
    const size_t id = lp_variables_.size();
    lp_variables_.push_back(std::make_shared<lp::LpVariable>(id, 0));
    lp_objective_.emplace_back(0);  // c_id
    return lp_variables_.at(id);
}

std::shared_ptr<lp::LpConstraint> lp::LpSolver::addConstraint() {
    const auto new_constraint = std::make_shared<lp::LpConstraint>();
    lp_constraints_.push_back(new_constraint);
    return new_constraint;
}

double lp::LpSolver::evaluateObjective(void) {
    // Slow, not made for using during solving
    double objective = 0.0;
    for (int i = 0; i < lp_variables_.size(); ++i) {
        objective += lp_objective_.at(i) * lp_variables_.at(i)->val;
    }
    return objective;
}

void lp::LpSolver::addToObjective(const lp::LpVariable& var,
                                  const double coeff) {
    if (var.getId() > lp_variables_.size()) {
        throw std::invalid_argument(std::format(
            "Incosistent variable id: {}, does it belong to another LP?",
            var.getId()));
    }
    lp_objective_.at(var.getId()) = coeff;
}

const int lp::LpSolver::numConstraintRows() const {
    const auto count_constraints = [](int sum,
                                      const std::pair<double, double> bounds) {
        return sum +
               int(bounds.first != std::numeric_limits<double>::lowest()) +
               int(bounds.second != std::numeric_limits<double>::max());
    };

    int n_constraints = std::accumulate(
        lp_constraints_.begin(), lp_constraints_.end(), 0,
        [&count_constraints](
            int sum, const std::shared_ptr<lp::LpConstraint> constraint) {
            return count_constraints(sum, constraint->bounds);
        });
    n_constraints += std::accumulate(
        lp_variables_.begin(), lp_variables_.end(), 0,
        [&count_constraints](int sum,
                             const std::shared_ptr<lp::LpVariable> variable) {
            return count_constraints(sum, variable->bounds);
        });
    return n_constraints;
}

template <typename T>
concept LpConstrainedType =
    std::is_same_v<T, lp::LpVariable> || std::is_same_v<T, lp::LpConstraint>;

// Correct function definition using constrained template parameter
template <LpConstrainedType T>
int appendRowsToMatrices(
    Eigen::MatrixXd& A, Eigen::MatrixXd& b,
    const std::vector<std::shared_ptr<T>>& constrained_type,
    const size_t n_variables, int n_added) {
    for (const auto ptr : constrained_type) {
        const auto rows = ptr->generateRows(n_variables);
        if (rows.rhs.rows()) {
            A.row(n_added) = rows.lhs.row(0);
            b.row(n_added) = rows.rhs.row(0);
            ++n_added;
            if (rows.rhs.rows() > 1) {
                A.row(n_added) = rows.lhs.row(1);
                b.row(n_added) = rows.rhs.row(1);
                ++n_added;
            }
        }
    }
    return n_added;
}

lp::MatrixContainer lp::LpSolver::getMatrices() {
    // Generate the matrix objects representing the constraints and the
    // objective

    // I will begin by generating rows for A and b from the constraints and
    // variables

    const size_t n_variables = lp_variables_.size();
    const size_t n_constraints = numConstraintRows();

    Eigen::MatrixXd A(n_constraints, n_variables);
    Eigen::MatrixXd b(n_constraints, 1);
    // Fill in with constraints first, then variables:
    int n_added = 0;
    n_added = appendRowsToMatrices(A, b, lp_constraints_, n_variables, n_added);
    n_added = appendRowsToMatrices(A, b, lp_variables_, n_variables, n_added);
    // TODO: finish! Will probably have to make lp_constraints_ into a vector of
    // shared_ptr.

    // Wrap A and b in shared pointers
    const auto A_ptr = std::make_shared<Eigen::MatrixXd>(A);
    const auto b_ptr = std::make_shared<Eigen::MatrixXd>(b);

    lp::MatrixContainer mc;
    mc.A = A_ptr;
    mc.b = b_ptr;

    return mc;
}

// Lp solver

// Lp constraint

void lp::LpConstraint::setBounds(const double lb, const double ub) {
    commonSetBounds(bounds, lb, ub);
}
void lp::LpConstraint::setLb(const double lb) {
    commonSetBounds(bounds, lb, {});
}
void lp::LpConstraint::setUb(const double ub) {
    commonSetBounds(bounds, {}, ub);
}

// Giving var as variable, just to not make it able to get an invalid id.
void lp::LpConstraint::addVariable(const double coeff,
                                   const std::shared_ptr<lp::LpVariable> var) {
    const size_t var_id = var->getId();
    const auto it = std::ranges::find(vars, var_id);
    const size_t idx = std::distance(vars.begin(), it);
    if (it != vars.end()) {
        // If the variable already exists, correct the current coefficient with
        // the provided one through addition. Could be better to either throw or
        // to replace, we will see...
        coeffs.at(idx) += coeff;
        utils::warn(
            std::format("Variable {} is already in constraint. Combined new "
                        "coefficient becomes: {}",
                        *it, coeffs.at(idx)));
    } else {
        // In the case the variable has not been added, add it.
        vars.push_back(var_id);
        coeffs.push_back(coeff);
    }
}

// A, b
const lp::ConstraintRows lp::LpConstraint::generateRows(
    const size_t num_variables) const {
    if (bounds.first == std::numeric_limits<double>::min() &&
        bounds.second == std::numeric_limits<double>::max()) {
        throw std::logic_error(
            "Unbounded constraints are not allowed. Make sure to "
            "set at least one bound.");
    }

    // Generate row
    Eigen::MatrixXd row(1, num_variables);
    for (size_t i = 0; i < vars.size(); ++i) {
        row(0, vars.at(i)) = coeffs.at(i);
    }
    return commonGenerateRows(row, bounds);
}

// Lp constraint

// Lp variable

void lp::LpVariable::setLb(const double lb) { commonSetBounds(bounds, lb, {}); }
void lp::LpVariable::setUb(const double ub) { commonSetBounds(bounds, {}, ub); }
void lp::LpVariable::setBounds(const double lb, const double ub) {
    commonSetBounds(bounds, lb, ub);
}

const lp::ConstraintRows lp::LpVariable::generateRows(
    const size_t num_variables) const {
    // Not throwing if unbounded, as that is fine with variables.

    // Warn if not on standard form
    if (bounds.first != 0.0) {
        utils::warn(std::format("Variable {} is not in standard form!", id));
    }

    // Generate row
    Eigen::MatrixXd row = Eigen::MatrixXd::Constant(1, num_variables, 0.0);
    row(0, id) = 1;  // As we are having constraints on the variables itself, it
                     // appears with coefficient 1
    return commonGenerateRows(row, bounds);
}

// Lp variable