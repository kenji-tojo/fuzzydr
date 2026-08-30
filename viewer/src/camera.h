// src/camera.h
#pragma once
#include <cmath>
#include <algorithm>

namespace fuzzydr_viewer {

/// Orbit camera:
///   Left-drag         : orbit   (drag up -> camera rises)
///   Right/middle-drag : pan     (follows cursor in camera plane)
///   Scroll            : zoom
///
/// Coordinate system: Z-up, right-handed.
/// All matrices are column-major (GLSL / Vulkan convention).
struct OrbitCamera {
    float center[3] = {0.f, 0.f, 0.f};
    float radius    = 5.f;
    float theta     = 1.5708f; // polar angle from +Z  (0 = top, pi/2 = equator, pi = bottom)
    float phi       = -1.5708f; // azimuth in XY plane
    float fovY      = 45.f;   // degrees
    float zNear     = 0.01f;
    float zFar      = 1000.f;

    /// Sign of the world-up axis (Z). +1 is the training convention; -1 points
    /// the camera's up reference the other way, which rotates the rendered
    /// image 180 degrees about the forward axis rather than mirroring it.
    /// Only the view matrix is affected; the scene is never transformed.
    int up_sign = +1;

    // ---- eye position ------------------------------------------------
    void eye(float out[3]) const {
        float st = std::sin(theta), ct = std::cos(theta);
        float sp = std::sin(phi),   cp = std::cos(phi);
        out[0] = center[0] + radius * st * cp;
        out[1] = center[1] + radius * st * sp;
        out[2] = center[2] + radius * ct;
    }

    // ---- input -------------------------------------------------------

    /// Orbit by (dTheta, dPhi) radians.
    /// Positive dTheta moves the camera downward (theta increases toward pi).
    /// Positive dPhi rotates counterclockwise when viewed from above.
    /// Both deltas are scaled by up_sign, so under a flipped up the drag
    /// direction still matches the screen: drag-up still raises the camera.
    void orbit(float dTheta, float dPhi) {
        const float s = float(up_sign);
        theta = std::clamp(theta + dTheta * s, 0.01f, 3.13f);
        phi  += dPhi * s;
    }

    /// Zoom by a multiplicative factor (< 1 = zoom in, > 1 = zoom out).
    void zoom(float factor) {
        radius = std::max(1e-4f, radius * factor);
    }

    /// Pan in the camera-local right/up plane.
    /// dx, dy are raw pixel deltas from the cursor; scale with radius so
    /// panning feels consistent regardless of zoom level.
    void pan(float dx, float dy) {
        float e[3];  eye(e);
        float f[3] = { center[0]-e[0], center[1]-e[1], center[2]-e[2] };
        float flen = std::sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);
        if (flen < 1e-7f) return;
        for (auto& x : f) x /= flen;

        float wUp[3] = {0.f, 0.f, float(up_sign)};
        if (std::abs(f[2]) > 0.99f) wUp[0] = 0.001f; // near-pole gimbal guard

        // right = f x wUp  (normalised)
        float r[3] = { f[1]*wUp[2]-f[2]*wUp[1],
                       f[2]*wUp[0]-f[0]*wUp[2],
                       f[0]*wUp[1]-f[1]*wUp[0] };
        float rlen = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
        if (rlen < 1e-7f) return;
        for (auto& x : r) x /= rlen;

        // up = r x f  (camera-local up)
        float u[3] = { r[1]*f[2]-r[2]*f[1],
                       r[2]*f[0]-r[0]*f[2],
                       r[0]*f[1]-r[1]*f[0] };

        float scale = radius * 0.001f;
        for (int i = 0; i < 3; i++)
            center[i] -= (r[i] * dx - u[i] * dy) * scale;
    }

    // ---- matrices (column-major, Vulkan clip space) ------------------

    void viewMatrix(float m[16]) const {
        float e[3];  eye(e);
        float f[3] = { center[0]-e[0], center[1]-e[1], center[2]-e[2] };
        float flen = std::sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);
        for (auto& x : f) x /= flen;

        float wUp[3] = {0.f, 0.f, float(up_sign)};
        if (std::abs(f[2]) > 0.99f) wUp[0] = 0.001f;

        float r[3] = { f[1]*wUp[2]-f[2]*wUp[1],
                       f[2]*wUp[0]-f[0]*wUp[2],
                       f[0]*wUp[1]-f[1]*wUp[0] };
        float rlen = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
        for (auto& x : r) x /= rlen;

        float u[3] = { r[1]*f[2]-r[2]*f[1],
                       r[2]*f[0]-r[0]*f[2],
                       r[0]*f[1]-r[1]*f[0] };

        // column-major: m[col*4 + row]
        m[ 0]= r[0]; m[ 4]= r[1]; m[ 8]= r[2]; m[12]=-(r[0]*e[0]+r[1]*e[1]+r[2]*e[2]);
        m[ 1]= u[0]; m[ 5]= u[1]; m[ 9]= u[2]; m[13]=-(u[0]*e[0]+u[1]*e[1]+u[2]*e[2]);
        m[ 2]=-f[0]; m[ 6]=-f[1]; m[10]=-f[2]; m[14]=  f[0]*e[0]+f[1]*e[1]+f[2]*e[2];
        m[ 3]= 0.f;  m[ 7]= 0.f;  m[11]= 0.f;  m[15]= 1.f;
    }

    /// Standard perspective, Y-flipped for Vulkan NDC.
    void projMatrix(float m[16], float aspect) const {
        float rad = fovY * 3.14159265f / 180.f;
        float f   = 1.f / std::tan(rad * 0.5f);
        float nf  = 1.f / (zNear - zFar);
        m[ 0]=f/aspect; m[ 4]=0.f; m[ 8]=0.f;       m[12]=0.f;
        m[ 1]=0.f;      m[ 5]=-f;  m[ 9]=0.f;       m[13]=0.f;  // flip Y
        m[ 2]=0.f;      m[ 6]=0.f; m[10]=zFar*nf;   m[14]=zFar*zNear*nf;
        m[ 3]=0.f;      m[ 7]=0.f; m[11]=-1.f;      m[15]=0.f;
    }

    /// Combined viewproj = proj * view (column-major).
    void viewprojMatrix(float vp[16], float aspect) const {
        float v[16], p[16];
        viewMatrix(v);
        projMatrix(p, aspect);
        for (int col = 0; col < 4; col++)
            for (int row = 0; row < 4; row++) {
                vp[col*4+row] = 0.f;
                for (int k = 0; k < 4; k++)
                    vp[col*4+row] += p[k*4+row] * v[col*4+k];
            }
    }

    // ---- auto-fit ----------------------------------------------------

    /// Fit the camera to the axis-aligned bounding box of the vertex positions.
    /// vert_attrs: float[num_verts * 7]  (x, y, z, radius, r, g, b)
    void autoFit(const float* vert_attrs, uint32_t num_verts) {
        if (num_verts == 0) return;
        float mn[3] = { 1e30f,  1e30f,  1e30f};
        float mx[3] = {-1e30f, -1e30f, -1e30f};
        for (uint32_t i = 0; i < num_verts; i++) {
            const float* v = vert_attrs + i * 7;
            for (int j = 0; j < 3; j++) {
                mn[j] = std::min(mn[j], v[j]);
                mx[j] = std::max(mx[j], v[j]);
            }
        }
        for (int j = 0; j < 3; j++) center[j] = (mn[j] + mx[j]) * 0.5f;
        float dx = mx[0]-mn[0], dy = mx[1]-mn[1], dz = mx[2]-mn[2];
        float size = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (size < 1e-7f) size = 1.f;
        radius = size * 1.5f;
        zNear  = size * 0.001f;
        zFar   = size * 20.f;
    }

    /// Initial orbit direction; call once up_sign is set.  up_sign flips which
    /// pole reads as up, so the elevation is signed to look slightly down
    /// under either convention.
    void setInitialOrbit() {
        theta = 1.5708f - float(up_sign) * 0.2792f; // 16 degrees above the horizon
        phi   = -2.376f;
    }
};

} // namespace fuzzydr_viewer
