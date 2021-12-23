//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToStd/TorchToStd.h"

#include "../PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// -----------------------------------------------------------------------------
// Patterns (as this grows, it should be organized into multiple files)
// -----------------------------------------------------------------------------
// This is going to eventually be O(#torch operators), which is in the 100s.

namespace {
// Note: Confusingly, ATen's "dim" means "number of dimensions" which is what
// MLIR calls "rank".
class ConvertAtenDimOp : public OpConversionPattern<AtenDimOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rank = rewriter.create<RankOp>(op->getLoc(), adaptor.self());
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(
        op, getTypeConverter()->convertType(op.getType()), rank);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenAddIntOp : public OpConversionPattern<AtenAddIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenAddIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, adaptor.a(), adaptor.b());
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenNeIntOp : public OpConversionPattern<AtenNeIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenNeIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, arith::CmpIPredicate::ne,
                                               adaptor.a(), adaptor.b());
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenGtIntOp : public OpConversionPattern<AtenGtIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenGtIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, arith::CmpIPredicate::sgt,
                                               adaptor.a(), adaptor.b());
    return success();
  }
};
} // namespace

namespace {
template <typename OpTy>
class ConvertTorchConstantOp : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.valueAttr());
    return success();
  }
};
} // namespace

namespace {
// for element-wise only
class ConvertExternOp : public OpConversionPattern<ExternOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ExternOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    op.emitRemark("in convert extern op");
	  Location loc = op->getLoc();
		MLIRContext *context = op->getContext();

	  // get element type
	  Type elementType = mlir::FloatType::getF32(context);

	  // get rank
	  int64_t inputRank = 2;

	  // get operand number
	  int64_t operandNum = 3;

    // get function name
		StringAttr name = adaptor.name();
		char *strName = (char*)name.data();
		op->emitRemark("adaptor.name: ") << strName;
		SmallVector<char *> nameParts;
		const char *delim = ".";
		char *p = strtok(strName, delim);
		while (nullptr != p) {
			nameParts.push_back(p);
			p = strtok(nullptr, delim);
		}
	  mlir::StringAttr libraryCallAttr = ::mlir::StringAttr::get(context, nameParts.back());
		op->emitRemark("library call attr: ") << libraryCallAttr;

		// get inputs
	  auto operands = adaptor.operands();
	  assert(operands.size() == operandNum && "Operand number error!");
	  Value input = operands[0];
//	  SmallVector<Value> inputs;
//    for (auto i : llvm::seq<int64_t>(0, operandNum))
//      inputs.push_back(operands[i]);

    // get result number
    int64_t resultNum = 1;

    // create type canonicalized memref operands
    SmallVector<Value, 4> callOperands;
	  callOperands.reserve(operands.size());

	  // inputs operands
	  for (auto op : operands) {
//	  	auto tensorType = op.getType().dyn_cast<TensorType>();
//	  	if (!tensorType) {
//			  callOperands.push_back(op);
//	  		continue;
//	  	}
//	  	Value cast = rewriter.create<tensor::CastOp>(loc, tensorType, op);
//		  callOperands.push_back(cast);
		  callOperands.push_back(op);
	  }

	  op->emitRemark("call operands: ");
	  for (auto v : callOperands) {
	  	v.dump();
	  }

	  // get function FuncOp
	  auto module = op->getParentOfType<ModuleOp>();
	  if (!module.lookupSymbol(libraryCallAttr)) {
		  // func input types
		  SmallVector<Type, 4> inputTypes;
		  inputTypes.reserve(operandNum);
		  for (auto operand : operands) {
		  	auto operandType = operand.getType();
			  inputTypes.push_back(operandType);
			  op->emitRemark("operandType");
			  operandType.dump();
		  }
		  // func output types
		  Type outputType = input.getType();
		  auto libFnType = rewriter.getFunctionType(inputTypes, outputType);

		  OpBuilder::InsertionGuard guard(rewriter);
		  // insert before module terminator.
		  rewriter.setInsertionPoint(module.getBody(),
															   std::prev(module.getBody()->end()));
		  FuncOp funcOp = rewriter.create<FuncOp>(loc, libraryCallAttr, libFnType);
		  funcOp->setAttr("llvm.emit_c_interface", UnitAttr::get(op->getContext()));
		  funcOp.setPrivate();
		  op->emitRemark("func op");
		  funcOp.dump();
	  }

    rewriter.replaceOpWithNewOp<mlir::CallOp>(op,
                                              libraryCallAttr.getValue(),
                                              input.getType(), callOperands);
	  op->emitRemark("convert extern success");
    return success();
  }
};
} // namespace

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToStd : public ConvertTorchToStdBase<ConvertTorchToStd> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<StandardOpsDialect>();
	  registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithmeticDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect, StandardOpsDialect,
                           arith::ArithmeticDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    target.addIllegalOp<AtenDimOp>();
    patterns.add<ConvertAtenDimOp>(typeConverter, context);
    target.addIllegalOp<AtenNeIntOp>();
    patterns.add<ConvertAtenNeIntOp>(typeConverter, context);
    target.addIllegalOp<AtenGtIntOp>();
    patterns.add<ConvertAtenGtIntOp>(typeConverter, context);
    target.addIllegalOp<ValueTensorLiteralOp>();
    patterns.add<ConvertTorchConstantOp<ValueTensorLiteralOp>>(typeConverter,
                                                               context);
    target.addIllegalOp<ConstantBoolOp>();
    patterns.add<ConvertTorchConstantOp<ConstantBoolOp>>(typeConverter,
                                                         context);
    target.addIllegalOp<Torch::ConstantFloatOp>();
    patterns.add<ConvertTorchConstantOp<Torch::ConstantFloatOp>>(typeConverter,
                                                                 context);
    target.addIllegalOp<Torch::ConstantIntOp>();
    patterns.add<ConvertTorchConstantOp<Torch::ConstantIntOp>>(typeConverter,
                                                               context);
    target.addIllegalOp<AtenAddIntOp>();
    patterns.add<ConvertAtenAddIntOp>(typeConverter, context);

    target.addIllegalOp<ExternOp>();
    patterns.add<ConvertExternOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::torch::createConvertTorchToStdPass() {
  return std::make_unique<ConvertTorchToStd>();
}
