// Copyright (c) Mixed Reality Toolkit Contributors
// Licensed under the BSD 3-Clause

// Ilias: 
// This script is the adapted version of an official MRTK3 script, named "HandJointVisualizer.cs" which can be found here: https://github.com/MixedRealityToolkit/MixedRealityToolkit-Unity/tree/main/org.mixedrealitytoolkit.input/Visualizers/HandJointVisualizer
// The MRTK names of the hand joints can be found here: https://learn.microsoft.com/en-us/dotnet/api/microsoft.mixedreality.toolkit.utilities.trackedhandjoint?view=mixed-reality-toolkit-unity-2020-dotnet-2.8.0

using MixedReality.Toolkit.Subsystems;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.XR;
using System.IO;

using TMPro;
using Unity.Barracuda;

namespace MixedReality.Toolkit.Input
{
    /// <summary>
    /// Basic hand joint visualizer that draws an instanced mesh on each hand joint.
    /// This visualizer is mostly recommended for debugging purposes; try the
    /// the <see cref="RiggedHandMeshVisualizer"/> for a more visually pleasing
    /// hand visualization.
    /// </summary>
    /// 
    /// [AddComponentMenu("MRTK/Input/Visualizers/Hand Joint Visualizer")]
    public class RightHandGesturesDetector : MonoBehaviour
    {
        [SerializeField]
        [Tooltip("The XRNode on which this hand is located.")]
        private XRNode handNode = XRNode.RightHand;

        /// <summary> The XRNode on which this hand is located. </summary>
        public XRNode HandNode { get => handNode; set => handNode = value; }

        [SerializeField]
        [Tooltip("Joint visualization mesh.")]
        private Mesh jointMesh;

        /// <summary> Joint visualization mesh. </summary>
        public Mesh JointMesh { get => jointMesh; set => jointMesh = value; }

        [SerializeField]
        [Tooltip("Joint visualization material.")]
        private Material jointMaterial;

        /// <summary> Joint visualization material. </summary>
        public Material JointMaterial { get => jointMaterial; set => jointMaterial = value; }

        private HandsAggregatorSubsystem handsSubsystem;

        // Transformation matrix for each joint.
        private List<Matrix4x4> jointMatrices = new List<Matrix4x4>();


        // Ilias
        // public GameObject sphereMarker;
        float timeStep = 0.02f;
        float timeInterval = 0.0f;
        List<Vector3[]> windowData = new List<Vector3[]>();
        int windowMaxSize = 11;
        float[] prediction = new float[2];
        bool insertDataToWindow = false;
        Vector3[] jointsDiff;
        Vector3[] currJoints, prevJoints;
        [SerializeField]
        private TMP_Text outputPrediction;
        [SerializeField]
        private NNModel onnxModel;
        private Model runtimeModel;
        private IWorker worker;
        private string outputLayerName;
        float predictionDisplayTimer = 0;

        private float time = 0.0f;


        /// <summary>
        /// A Unity event function that is called when the script component has been enabled.
        /// </summary>
        protected void OnEnable()
        {
            Debug.Assert(handNode == XRNode.LeftHand || handNode == XRNode.RightHand, $"HandVisualizer has an invalid XRNode ({handNode})!");

            handsSubsystem = XRSubsystemHelpers.GetFirstRunningSubsystem<HandsAggregatorSubsystem>();

            if (handsSubsystem == null)
            {
                StartCoroutine(EnableWhenSubsystemAvailable());
            }
            else
            {
                for (int i = 0; i < (int)TrackedHandJoint.TotalJoints; i++)
                {
                    jointMatrices.Add(new Matrix4x4());
                }
            }

            runtimeModel = ModelLoader.Load(onnxModel);
            worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, runtimeModel);
            outputLayerName = runtimeModel.outputs[runtimeModel.outputs.Count - 1];
        }

        /// <summary>
        /// A Unity event function that is called to draw Unity editor gizmos that are also interactable and always drawn.
        /// </summary>
        private void OnDrawGizmos()
        {
            if (!enabled) { return; }

            // Query all joints in the hand.
            if (handsSubsystem == null || !handsSubsystem.TryGetEntireHand(handNode, out IReadOnlyList<HandJointPose> joints))
            {
                return;
            }

            for (int i = 0; i < joints.Count; i++)
            {
                HandJointPose joint = joints[i];
                Gizmos.color = Color.blue;
                Gizmos.DrawRay(joint.Position, joint.Forward * 0.01f);
                Gizmos.color = Color.red;
                Gizmos.DrawRay(joint.Position, joint.Right * 0.01f);
                Gizmos.color = Color.green;
                Gizmos.DrawRay(joint.Position, joint.Up * 0.01f);
            }
        }

        /// <summary>
        /// Coroutine to wait until subsystem becomes available.
        /// </summary>
        private IEnumerator EnableWhenSubsystemAvailable()
        {
            yield return new WaitUntil(() => XRSubsystemHelpers.GetFirstRunningSubsystem<HandsAggregatorSubsystem>() != null);
            OnEnable();
        }

        /// <summary>
        /// A Unity event function that is called every frame, if this object is enabled.
        /// </summary>
        private void Update()
        {
            // Clear the prediction sign after a while. 
            if (predictionDisplayTimer > 0.0f)
            {
                predictionDisplayTimer -= Time.unscaledDeltaTime;
                if (predictionDisplayTimer < 0.0f)
                {
                    outputPrediction.text = "";
                    predictionDisplayTimer = 0;
                }
            }

            // Query all joints in the hand.
            if (handsSubsystem == null || !handsSubsystem.TryGetEntireHand(handNode, out IReadOnlyList<HandJointPose> joints))
            {
                windowData.Clear();
                insertDataToWindow = false;
                return;
            }

            // sphereMarker.transform.position = joints[(int)TrackedHandJoint.ThumbTip].Position;

            RenderJoints(joints);
            ProcessFrame(joints);

        }

        private void RenderJoints(IReadOnlyList<HandJointPose> joints)
        {
            for (int i = 0; i < joints.Count; i++)
            {
                // Skip joints with uninitialized quaternions.
                // This is temporary; eventually the HandsSubsystem will
                // be robust enough to never give us broken joints.
                if (joints[i].Rotation.Equals(new Quaternion(0, 0, 0, 0)))
                {
                    continue;
                }

                // Fill the matrices list with TRSs from the joint poses.
                jointMatrices[i] = Matrix4x4.TRS(joints[i].Position, joints[i].Rotation.normalized, Vector3.one * joints[i].Radius);
            }

            // Draw the joints.
            Graphics.DrawMeshInstanced(jointMesh, 0, jointMaterial, jointMatrices);
        }

        private void ProcessFrame(IReadOnlyList<HandJointPose> joints)
        {
            // Debug.Log("UnscaledDeltaTime: " + Time.unscaledDeltaTime);
            timeInterval += Time.unscaledDeltaTime;
            if (timeInterval < timeStep)
            {
                return;
            }
            else
            {
                // Debug.Log("Interval: " + timeInterval);
                // timeInterval = 0f;
                timeInterval = timeInterval - timeStep;
            }

            // Debug.Log("Time: " + (Time.time - time));
            // time = Time.time;

            // Debug.Log(joints.Count);

            if (insertDataToWindow == false)
            {
                prevJoints = joints.Select(c => c.Position).ToArray();
                insertDataToWindow = true;
                return;
            }
            else
            {
                currJoints = joints.Select(c => c.Position).ToArray();
                jointsDiff = subtractVector3Arrays(currJoints, prevJoints);
                prevJoints = currJoints;
                windowData.Insert(0, jointsDiff);
                if (windowData.Count == windowMaxSize)
                {
                    // Debug.Log(windowData.Count);
                    prediction = Predict(FlattenWindowData());
                    // Debug.Log("Random gest:" + prediction[0]);
                    // Debug.Log("NotOk:" + prediction[1]);
                    // if (prediction[0] > 0.1)
                    // {
                    //     outputPrediction.text = "RANDOM GEST !!";
                    //     predictionDisplayTimer = 0.4f;
                    // }
                    if (prediction[1] > 0.6)
                    {
                        Debug.Log("NotOk:" + prediction[1]);
                        outputPrediction.text = "NOT_OK GEST !!";
                        predictionDisplayTimer = 2f;
                        windowData.Clear();
                        insertDataToWindow = false;
                    }
                    else
                    {
                        windowData.RemoveAt(windowData.Count - 1);
                    }
                }
            }
        }

        private float[] Predict(float[] data)
        {
            using Tensor inputTensor = new Tensor(1, 858, data);
            worker.Execute(inputTensor);
            Tensor outputTensor = worker.PeekOutput(outputLayerName);

            float[] output = new float[2];
            // Debug.Log(outputTensor.length);

            // The network outputs two values: The first one is the chance of the gesture to be a random gesture and the second one is the chance to be a 'notOk' gesture.
            for (int i = 0; i < 2; i++)
            {
                output[i] = outputTensor[i];
            }

            // Clean up
            inputTensor.Dispose();
            outputTensor.Dispose();

            return output;
        }

        private float[] FlattenWindowData()
        {
            float[] flattened = new float[858];
            for (int i = 0; i < 11; i++)
            {
                for (int j = 0; j < 26; j++)
                {
                    flattened[i * 26 + j * 3] = windowData[i][j].x;
                    flattened[i * 26 + j * 3 + 1] = windowData[i][j].y;
                    flattened[i * 26 + j * 3 + 2] = windowData[i][j].z;
                }

                // for (int j = 0; j < 26; j++)
                // {
                //     flattened[i * 26 + j * 3] = 0.01f;
                //     flattened[i * 26 + j * 3 + 1] = 0.01f;
                //     flattened[i * 26 + j * 3 + 2] = 0.01f;
                // }
            }
            return flattened;
        }

        private Vector3[] subtractVector3Arrays(Vector3[] a, Vector3[] b)
        {
            Vector3[] c = new Vector3[26];
            for (int i = 0; i < 26; i++)
            {
                c[i] = a[i] - b[i];
            }
            return c;
        }

        private void OnDestroy()
        {
            worker?.Dispose();
        }

    }
}
