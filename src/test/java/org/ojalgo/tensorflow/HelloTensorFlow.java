/*
 * Copyright 1997-2021 Optimatika
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package org.ojalgo.tensorflow;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.ojalgo.TestUtils;
import org.ojalgo.function.constant.PrimitiveMath;
import org.tensorflow.ConcreteFunction;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Signature;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.math.Add;
import org.tensorflow.proto.framework.DataType;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;

public class HelloTensorFlow {

    @Test
    public void testFirstHello() throws Exception {

        try (ConcreteFunction function = ConcreteFunction.create(HelloTensorFlow::doubler)) {

            try (TInt32 x = TInt32.scalarOf(10); Tensor y = function.call(x)) {

                TestUtils.assertEquals(Shape.scalar(), y.shape());
                TestUtils.assertEquals(0, y.rank());
                TestUtils.assertEquals(1, y.size());

                TestUtils.assertEquals(DataType.DT_INT32, y.dataType());

                TestUtils.assertEquals(20, ((TInt32) y).getInt());
            }

            try (TInt32 x = TInt32.vectorOf(0, 1, 2, 3, 4); Tensor y = function.call(x)) {

                TestUtils.assertEquals(Shape.of(5), y.shape());
                TestUtils.assertEquals(1, y.rank());
                TestUtils.assertEquals(5, y.size());

                TestUtils.assertEquals(DataType.DT_INT32, y.dataType());

                for (int i = 0; i < 5; i++) {
                    TestUtils.assertEquals(2 * i, ((TInt32) y).getInt(i));
                }
            }
        }
    }

    private static Signature doubler(final Ops tf) {
        Placeholder<TInt32> x = tf.placeholder(TInt32.class);
        Add<TInt32> dblX = tf.math.add(x, x);
        return Signature.builder().input("x", x).output("dbl", dblX).build();
    }

    @Test
    public void testBasicGraph() throws Exception {

        try (Graph graph = new Graph(); TFloat64 _3 = TFloat64.scalarOf(3.0); TFloat64 _2 = TFloat64.scalarOf(2.0); TFloat64 _6 = TFloat64.scalarOf(6.0);) {

            Operation a = graph.opBuilder("Const", "a").setAttr("dtype", DataType.DT_DOUBLE).setAttr("value", _3).build();
            Operation b = graph.opBuilder("Const", "b").setAttr("dtype", DataType.DT_DOUBLE).setAttr("value", _2).build();

            Operation x = graph.opBuilder("Placeholder", "x").setAttr("dtype", DataType.DT_DOUBLE).build();
            Operation y = graph.opBuilder("Placeholder", "y").setAttr("dtype", DataType.DT_DOUBLE).build();

            Operation ax = graph.opBuilder("Mul", "ax").addInput(a.output(0)).addInput(x.output(0)).build();
            Operation by = graph.opBuilder("Mul", "by").addInput(b.output(0)).addInput(y.output(0)).build();
            Operation z = graph.opBuilder("Add", "z").addInput(ax.output(0)).addInput(by.output(0)).build();

            try (Session session = new Session(graph)) {

                List<Tensor> output = session.runner().fetch("z").feed("x", _3).feed("y", _6).run();

                TestUtils.assertEquals(21.0, ((TFloat64) output.get(0)).getDouble(), PrimitiveMath.MACHINE_EPSILON);
            }
        }

    }
}
